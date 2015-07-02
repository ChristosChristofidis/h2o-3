package water.currents;

import water.H2O;
import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;
import water.fvec.Vec;
import water.parser.ValueString;

/**
 * Subclasses auto-widen between scalars and Frames, and have exactly two arguments
 */
abstract class ASTBinOp extends ASTPrim {
  @Override int nargs() { return 1+2; }
  @Override Val apply( Env env, Env.StackHelp stk, AST asts[] ) {
    Val left = stk.track(asts[1].exec(env));
    Val rite = stk.track(asts[2].exec(env));
    return prim_apply(left,rite);
  }

  Val prim_apply( Val left, Val rite ) {
    switch( left.type() ) {
    case Val.NUM: 
      final double dlf = left.getNum();
      
      switch( rite.type() ) {
      case Val.NUM:  return new ValNum( op (dlf,rite.getNum()));
      case Val.FRM:  return scalar_op_frame(dlf,rite.getFrame());
      case Val.STR:  throw H2O.unimpl();
      default: throw H2O.fail();
      }

    case Val.FRM: 
      Frame flf = left.getFrame();

      switch( rite.type() ) {
      case Val.NUM:  return frame_op_scalar(flf,rite.getNum());
      case Val.STR:  return frame_op_scalar(flf,rite.getStr());
      case Val.FRM:  return frame_op_frame (flf,rite.getFrame());
      default: throw H2O.fail();
      }
          
    case Val.STR: throw H2O.unimpl();
    default: throw H2O.fail();
    }
  }
  /** Override to express a basic math primitive */
  abstract double op( double l, double r );
  double str_op( ValueString l, ValueString r ) { throw H2O.fail(); }

  /** Auto-widen the scalar to every element of the frame */
  private ValFrame scalar_op_frame( final double d, Frame fr ) {
    for( Vec vec : fr.vecs() )
      if( !vec.isNumeric() ) throw new IllegalArgumentException("Cannot mix Numeric and non-Numeric types");
    return new ValFrame(new MRTask() {
        @Override public void map( Chunk[] chks, NewChunk[] cress ) {
          for( int c=0; c<chks.length; c++ ) {
            Chunk chk = chks[c];
            NewChunk cres = cress[c];
            for( int i=0; i<chk._len; i++ )
              cres.addNum(op(d,chk.atd(i)));
          }
        }
      }.doAll(fr.numCols(),fr).outputFrame());
  }

  /** Auto-widen the scalar to every element of the frame */
  ValFrame frame_op_scalar( Frame fr, final double d ) {
    for( Vec vec : fr.vecs() )
      if( !vec.isNumeric() ) throw new IllegalArgumentException("Cannot mix Numeric and non-Numeric types");
    return new ValFrame(new MRTask() {
        @Override public void map( Chunk[] chks, NewChunk[] cress ) {
          for( int c=0; c<chks.length; c++ ) {
            Chunk chk = chks[c];
            NewChunk cres = cress[c];
            for( int i=0; i<chk._len; i++ )
              cres.addNum(op(chk.atd(i),d));
          }
        }
      }.doAll(fr.numCols(),fr).outputFrame());
  }

  /** Auto-widen the scalar to every element of the frame */
  private ValFrame frame_op_scalar( Frame fr, String str ) {
    for( Vec vec : fr.vecs() )
      if( !vec.isString() ) throw new IllegalArgumentException("Cannot mix String and non-String types");
    final ValueString srt = new ValueString(str);
    return new ValFrame(new MRTask() {
        @Override public void map( Chunk[] chks, NewChunk[] cress ) {
          ValueString vstr = new ValueString();
          for( int c=0; c<chks.length; c++ ) {
            Chunk chk = chks[c];
            NewChunk cres = cress[c];
            for( int i=0; i<chk._len; i++ )
              cres.addNum(str_op(chk.atStr(vstr,i),srt));
          }
        }
      }.doAll(1,fr).outputFrame());
  }

  /** Auto-widen: If one frame has only 1 column, auto-widen that 1 column to
   *  the rest.  Otherwise the frames must have the same column count, and
   *  auto-widen element-by-element.  Short-cut if one frame has zero
   *  columns. */
  private ValFrame frame_op_frame( Frame lf, Frame rt ) {
    if( lf.numRows() != rt.numRows() ) 
      throw new IllegalArgumentException("Frames must have same rows, found "+lf.numRows()+" rows and "+rt.numRows()+" rows.");
    if( lf.numCols() == 0 ) return new ValFrame(lf);
    if( rt.numCols() == 0 ) return new ValFrame(rt);
    if( lf.numCols() == 1 && rt.numCols() > 1 ) return vec_op_frame(lf.vecs()[0],rt);
    if( rt.numCols() == 1 && lf.numCols() > 1 ) return frame_op_vec(lf,rt.vecs()[0]);
    if( lf.numCols() != rt.numCols() )
      throw new IllegalArgumentException("Frames must have same columns, found "+lf.numCols()+" columns and "+rt.numCols()+" columns.");

    return new ValFrame(new MRTask() {
        @Override public void map( Chunk[] chks, NewChunk[] cress ) {
          assert (cress.length<<1) == chks.length;
          for( int c=0; c<cress.length; c++ ) {
            Chunk clf = chks[c];
            Chunk crt = chks[c+cress.length];
            NewChunk cres = cress[c];
            for( int i=0; i<clf._len; i++ )
              cres.addNum(op(clf.atd(i),crt.atd(i)));
          }
        }
      }.doAll(lf.numCols(),new Frame(lf).add(rt)).outputFrame());
  }

  private ValFrame vec_op_frame( Vec vec, Frame fr ) {
    throw H2O.unimpl();
  }
  private ValFrame frame_op_vec( Frame fr, Vec vec ) {
    throw H2O.unimpl();
  }
}

// ----------------------------------------------------------------------------
// Expressions that auto-widen between NUM and FRM
class ASTAnd  extends ASTBinOp { String str() { return "&" ; } double op( double l, double r ) { return ASTLAnd.and_op(l,r); } }
class ASTDiv  extends ASTBinOp { String str() { return "/" ; } double op( double l, double r ) { return l/ r; } }
class ASTMul  extends ASTBinOp { String str() { return "*" ; } double op( double l, double r ) { return l* r; } }
class ASTOr   extends ASTBinOp { String str() { return "|" ; } double op( double l, double r ) { return ASTLOr . or_op(l,r); } }
class ASTPlus extends ASTBinOp { String str() { return "+" ; } double op( double l, double r ) { return l+ r; } }
class ASTPow  extends ASTBinOp { String str() { return "^" ; } double op( double l, double r ) { return Math.pow(l,r); } }
class ASTSub  extends ASTBinOp { String str() { return "-" ; } double op( double l, double r ) { return l- r; } }

class ASTRound extends ASTBinOp { 
  String str() { return "round"; } 
  double op(double x, double digits) { 
    // e.g.: floor(2.676*100 + 0.5) / 100 => 2.68
    if(Double.isNaN(x)) return x;
    double sgn = x < 0 ? -1 : 1;
    x = Math.abs(x);
    double power_of_10 = (int)Math.pow(10, (int)digits);
    return sgn*(digits == 0
                // go to the even digit
                ? (x % 1 >= 0.5 && !(Math.floor(x)%2==0))
                ? Math.ceil(x)
                : Math.floor(x)
                : Math.floor(x * power_of_10 + 0.5) / power_of_10);
  }
}

class ASTSignif extends ASTBinOp { 
  String str() { return "signif"; } 
  double op(double x, double digits) { 
    if(Double.isNaN(x)) return x;
    java.math.BigDecimal bd = new java.math.BigDecimal(x);
    bd = bd.round(new java.math.MathContext((int)digits, java.math.RoundingMode.HALF_EVEN));
    return bd.doubleValue();
  }
}


class ASTGE   extends ASTBinOp { String str() { return ">="; } double op( double l, double r ) { return l>=r?1:0; } }
class ASTGT   extends ASTBinOp { String str() { return ">" ; } double op( double l, double r ) { return l> r?1:0; } }
class ASTLE   extends ASTBinOp { String str() { return "<="; } double op( double l, double r ) { return l<=r?1:0; } }
class ASTLT   extends ASTBinOp { String str() { return "<" ; } double op( double l, double r ) { return l< r?1:0; } }

class ASTEQ   extends ASTBinOp { String str() { return "=="; } double op( double l, double r ) { return l==r?1:0; } 
  double str_op( ValueString l, ValueString r ) { return l==null ? (r==null?1:0) : (l.equals(r) ? 1 : 0); } 
  @Override ValFrame frame_op_scalar( Frame fr, final double d ) {
    return new ValFrame(new MRTask() {
        @Override public void map( Chunk[] chks, NewChunk[] cress ) {
          for( int c=0; c<chks.length; c++ ) {
            Chunk chk = chks[c];
            NewChunk cres = cress[c];
            if( !chk.vec().isNumeric() ) cres.addZeros(chk._len);
            else 
              for( int i=0; i<chk._len; i++ )
                cres.addNum(op(chk.atd(i),d));
          }
        }
      }.doAll(fr.numCols(),fr).outputFrame());
  }
}
class ASTNE   extends ASTBinOp { String str() { return "!="; } double op( double l, double r ) { return l!=r?1:0; } 
  double str_op( ValueString l, ValueString r ) { return l==null ? (r==null?0:1) : (l.equals(r) ? 0 : 1); } }

// ----------------------------------------------------------------------------
// Logical-AND.  If the first arg is false, do not execute the 2nd arg.
class ASTLAnd extends ASTBinOp { 
  String str() { return "&&"; } 
  @Override Val apply( Env env, Env.StackHelp stk, AST asts[] ) {
    Val left = stk.track(asts[1].exec(env));
    // If the left is zero or NA, do not evaluate the right, just return the left
    if( left.isNum() ) {
      double d = ((ValNum)left)._d;
      if( d==0 || Double.isNaN(d) ) return left;
    }
    Val rite = stk.track(asts[2].exec(env));
    return prim_apply(left,rite);
  }
  // Weird R semantics, zero trumps NA
  double op( double l, double r ) { return and_op(l,r); }
  static double and_op( double l, double r ) {   
    return (l==0||r==0) ? 0 : (Double.isNaN(l) || Double.isNaN(r) ? Double.NaN : 1); 
  } 
}

// Logical-OR.  If the first arg is true, do not execute the 2nd arg.
class ASTLOr extends ASTBinOp { 
  String str() { return "||"; } 
  @Override Val apply( Env env, Env.StackHelp stk, AST asts[] ) {
    Val left = stk.track(asts[1].exec(env));
    // If the left is non-zero or NA, do not evaluate the right, just return the left
    if( left.isNum() ) {
      double d = ((ValNum)left)._d;
      if( d!=0 || Double.isNaN(d) ) return left;
    }
    Val rite = stk.track(asts[2].exec(env));
    return prim_apply(left,rite);
  }
  // Weird R semantics, zero trumps NA
  double op( double l, double r ) { return or_op(l,r); }
  static double or_op( double l, double r ) {   
    return (l!=0||r!=0) ? 1 : (Double.isNaN(l) || Double.isNaN(r) ? Double.NaN : 0); 
  } 
}

// IfElse.  Execute either 2nd or 3rd (but not both) according to 1st arg.
// Neither if test is NaN.  Returns something of the same "shape" as the test -
// scalar for scalar, frame for frame.  Elements of the result are picked from
// either the true or false frame.
class ASTIfElse extends ASTPrim { 
  @Override int nargs() { return 1+3; } // test true false
  String str() { return "ifelse"; } 
  @Override Val apply( Env env, Env.StackHelp stk, AST asts[] ) {
    Val val = stk.track(asts[1].exec(env));

    if( val.isNum() ) {         // Scalar test, scalar result
      double d = val.getNum();
      if( Double.isNaN(d) ) return new ValNum(Double.NaN);
      Val res = stk.track(asts[d==0 ? 3 : 2].exec(env)); // exec only 1 of false and true
      return res.isFrame() ? new ValNum(res.getFrame().vec(0).at(0)) : res;
    }

    // Frame test.  Frame result.
    Frame tst = val.getFrame();

    // If all zero's, return false and never execute true.
    Frame fr = new Frame(tst);
    Frame tfr = null;
    for( Vec vec : tst.vecs() )
      if( vec.min()!=0 || vec.max()!= 0 ) {
        fr.add(tfr = exec_check(env,stk,tst,asts[2])); // Need true side sometimes
        break;
      }
    final boolean has_t = tfr!=null;

    // If all nonzero's (or NA's), then never execute false.
    Frame ffr = null;
    for( Vec vec : tst.vecs() )
      if( vec.nzCnt()+vec.naCnt() < vec.length() ) {
        fr.add(ffr = exec_check(env,stk,tst,asts[3])); // Need false side sometimes
        break;
      }
    final boolean has_f = ffr!=null;

    // Now pick from left-or-right in the new frame
    assert tfr != null || ffr != null;
    Frame res = new MRTask() {
        @Override public void map( Chunk chks[], NewChunk nchks[] ) {
          assert nchks.length+(has_t ? nchks.length : 0)+(has_f ? nchks.length : 0) == chks.length;
          for( int i=0; i<nchks.length; i++ ) {
            Chunk ctst = chks[i];
            NewChunk res = nchks[i];
            for( int row=0; row<ctst._len; row++ ) {
              double d = ctst.isNA(row) 
                ? Double.NaN    // Not from either side
                : chks[i+nchks.length+((ctst.atd(row)==0 && has_t) ? nchks.length : 0)].atd(row);
              res.addNum(d);
            }
          }
        }
      }.doAll(tst.numCols(),fr).outputFrame();
    return new ValFrame(res);
  }

  Frame exec_check( Env env, Env.StackHelp stk, Frame tst, AST ast ) {
    Frame fr = stk.track(ast.exec(env)).getFrame();
    if( tst.numCols() != fr.numCols() || tst.numRows() != fr.numRows() )
      throw new IllegalArgumentException("ifelse test frame and other frames must match dimensions, found "+tst+" and "+fr);
    return fr;
  }
}

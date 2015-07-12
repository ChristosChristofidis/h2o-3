package hex.deeplearning;

import hex.Distribution;
import hex.Grid;
import hex.Model;
import water.DKV;
import water.H2O;
import water.Key;
import water.fvec.Frame;
import hex.deeplearning.DeepLearningParameters.*;

import java.util.Map;

/** Grid for deep learning.
 */
public class DeepLearningGrid extends Grid<DeepLearningParameters, DeepLearningGrid> {

  public static final String MODEL_NAME = "DeepLearning";

  @Override protected String modelName() { return MODEL_NAME; }

  private static final String[] HYPER_NAMES    = new String[] { "activation", "epochs", "loss", "distribution"};
  private static final double[] HYPER_DEFAULTS = new double[] { Activation.Rectifier.ordinal(), 10,  Loss.Automatic.ordinal(), Distribution.Family.AUTO.ordinal() };

  @Override protected String[] hyperNames() { return HYPER_NAMES; }

  @Override protected double[] hyperDefaults() { return HYPER_DEFAULTS; }

  @Override protected double suggestedNextHyperValue( int h, Model m, double[] hyperLimits ) {
    throw H2O.unimpl();
  }

  @Override protected DeepLearning createBuilder(DeepLearningParameters params) {
    return new DeepLearning(params);
  }

  @Override
  protected DeepLearningParameters applyHypers(DeepLearningParameters parms, double[] hypers) {
    parms._activation = Activation.values()[(int) hypers[0]];
    parms._epochs = (int) hypers[1];
    parms._loss = Loss.values()[(int) hypers[2]];
    parms._distribution = Distribution.Family.values()[(int) hypers[3]];
    return parms;
  }

  @Override public double[] getHypers(DeepLearningParameters params) {
    double[] hypers = new double[HYPER_NAMES.length];
    hypers[0] = params._activation.ordinal();
    hypers[1] = params._epochs;
    hypers[2] = params._loss.ordinal();
    hypers[3] = params._distribution.ordinal();
    return hypers;
  }

  // Factory for returning a grid based on an algorithm flavor
  private DeepLearningGrid(Key key, Frame fr, DeepLearningParameters params, String[] hyperNames) {
    super(key, fr, params, hyperNames);
  }

  public static DeepLearningGrid get(Key<Grid> destKey, Frame fr, DeepLearningParameters params, Map<String,Object[]> hyperParams) {
    Key k = destKey != null ? destKey : Grid.keyName(MODEL_NAME, fr);
    DeepLearningGrid kmg = DKV.getGet(k);
    if( kmg != null ) return kmg;
    kmg = new DeepLearningGrid(k, fr, params, hyperParams.keySet().toArray(new String[hyperParams.size()]));
    DKV.put(kmg);
    return kmg;
  }
  /** FIXME: Rest API requirement - do not call directly */
  public DeepLearningGrid() { super(null, null, null, null); }
}

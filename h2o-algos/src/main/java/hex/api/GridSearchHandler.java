package hex.api;

import hex.Grid;
import hex.ModelParametersBuilderFactory;
import hex.schemas.GridSearchSchema;
import hex.Model;
import water.Job;
import water.Key;
import water.api.*;
import water.exceptions.H2OIllegalArgumentException;
import water.fvec.Frame;
import water.util.PojoUtils;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * FIXME: how to get rid of P, since it is already enforced by S
 *
 * @param <G>  Implementation output of grid search
 * @param <P>  Provides additional type information about grid search parameters.
 * @param <S>  Input/output schema produced by this grid search handler (IN: parameters, OUT: parameters + job)
 */
public abstract class GridSearchHandler< G extends Grid<MP, G>,
                                S extends GridSearchSchema<G,S, MP, P>,
                                MP extends Model.Parameters,
                                P extends ModelParametersSchema> extends Handler {

  public S do_train(int version, S gridSearchSchema) { // just following convention of model builders
    // Extract input parameters
    P parametersSchema = gridSearchSchema.parameters;
    // TODO: Verify algorithm inputs, make sure to reject wrong training_frame
    // Extract hyper parameters
    Map<String,Object[]> hyperParams = gridSearchSchema.hyper_parameters;
    // Verify list of hyper parameters
    // Right now only names, no types
    validateHyperParams(parametersSchema, hyperParams);

    // Get/create a grid for given frame
    Key<Grid> destKey = gridSearchSchema.grid_id != null ? gridSearchSchema.grid_id.key() : null;
    // Get actual parameters
    MP params = (MP) parametersSchema.createAndFillImpl();
    // Create target grid search object (keep it private for now)
    G grid = gridSearchSchema.fillImpl(
            createGrid(destKey, parametersSchema.training_frame.key().get(), params, hyperParams));
    // Start grid search and return the schema back with job key
    Grid.GridSearch gsJob = grid.startGridSearch(params, hyperParams);

    // Fill schema with job parameters
    // FIXME: right now we have to remove grid parameters which we sent back
    gridSearchSchema.hyper_parameters = null;
    gridSearchSchema.total_models = gsJob.getModelsCount();
    gridSearchSchema.job = (JobV3) Schema.schema(version, Job.class).fillFromImpl(gsJob);

    return gridSearchSchema;
  }

  // Force underlying handlers to create their grid implementations
  // - In the most of cases the call needs to be forwarded to GridSearch factory
  protected abstract G createGrid(Key<Grid> destKey, Frame f, MP params, Map<String,Object[]> hyperParams);

  /** Validate given hyper parameters with respect to type parameter P.
   *
   * It verifies that given parameters are annotated in P with @API annotation
   * @param params  regular model build parameters
   * @param hyperParams map of hyper parameters
   */
  protected void validateHyperParams(P params, Map<String, Object[]> hyperParams) {
    List<SchemaMetadata.FieldMetadata> fsMeta = SchemaMetadata.getFieldMetadata(params);
    for (Map.Entry<String, Object[]> hparam : hyperParams.entrySet()) {
      SchemaMetadata.FieldMetadata fieldMetadata = null;
      // Found corresponding metadata about the field
      for (SchemaMetadata.FieldMetadata fm : fsMeta) {
        if (fm.name.equals(hparam.getKey())) {
          fieldMetadata = fm;
          break;
        }
      }
      if (fieldMetadata == null)
        throw new H2OIllegalArgumentException(hparam.getKey(), "grid", "Unknown hyper parameter for grid search!");
      if (!fieldMetadata.is_gridable)
        throw new H2OIllegalArgumentException(hparam.getKey(), "grid",
                "Illegal hyper parameter for grid search! The parameter '" + fieldMetadata.name + " is not gridable!");
    }
  }


  static class DefaultModelParametersBuilderFactory<MP extends Model.Parameters,PS extends ModelParametersSchema>
          implements ModelParametersBuilderFactory<MP> {

    @Override
    public ModelParametersBuilder<MP> get(MP initialParams) {
      return new ModelParametersFromSchemaBuilder<MP, PS>(initialParams);
    }
  }

  public static class ModelParametersFromSchemaBuilder<MP extends Model.Parameters,PS extends ModelParametersSchema>
        implements ModelParametersBuilderFactory.ModelParametersBuilder<MP> {

    private MP params;
    private PS paramsSchema;
    private ArrayList<String> fields;

    public ModelParametersFromSchemaBuilder(MP initialParams) {
      params = initialParams;
      paramsSchema = (PS) Schema.schema(Schema.getHighestSupportedVersion(), params.getClass());
      fields = new ArrayList<>(7);
    }

    public ModelParametersFromSchemaBuilder<MP, PS> set(String name, Object value) {
      try {
        Field f = paramsSchema.getClass().getDeclaredField(name);
        API api = (API) f.getAnnotations()[0];
        Schema.setField(paramsSchema, f, name, value.toString(), api.required(), paramsSchema.getClass());
        fields.add(name);
      } catch (NoSuchFieldException e) {
        throw new IllegalArgumentException("Cannot find field '"+name+"'", e);
      } catch (IllegalAccessException e) {
        throw new IllegalArgumentException("Cannot set field '"+name+"'", e);
      }
      return this;
    }

    public MP build() {
      PojoUtils.copyProperties(params, paramsSchema, PojoUtils.FieldNaming.DEST_HAS_UNDERSCORES, null, fields.toArray(new String[fields.size()]));
      return params;
    }
  }
}






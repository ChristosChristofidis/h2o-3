import sys
sys.path.insert(1, "../../../")
import h2o
import pandas as pd
import zipfile
import statsmodels.api as sm

def link_functions_poisson(ip,port):
    # Connect to h2o
    h2o.init(ip,port)

    print("Read in prostate data.")
    h2o_data = h2o.import_frame(path=h2o.locate("smalldata/prostate/prostate_complete.csv.zip"))

    sm_data = pd.read_csv(zipfile.ZipFile(h2o.locate("smalldata/prostate/prostate_complete.csv.zip")).
                          open("prostate_complete.csv")).as_matrix()
    sm_data_response = sm_data[:,9]
    sm_data_features = sm_data[:,1:9]

    print("Testing for family: POISSON")
    print("Set variables for h2o.")
    myY = "GLEASON"
    myX = ["ID","AGE","RACE","CAPSULE","DCAPS","PSA","VOL","DPROS"]

    print("Create h2o model with canonical link: LOG")
    h2o_model_log = h2o.glm(x=h2o_data[myX], y=h2o_data[myY], family="poisson", link="log",alpha=[0.5], Lambda=[0])

    print("Create statsmodel model with canonical link: LOG")
    sm_model_log = sm.GLM(endog=sm_data_response, exog=sm_data_features,
                          family=sm.families.Poisson(sm.families.links.log)).fit()

    print("Compare model deviances for link function log")
    h2o_deviance_log = h2o_model_log.residual_deviance() / h2o_model_log.null_deviance()
    sm_deviance_log = sm_model_log.deviance / sm_model_log.null_deviance
    assert h2o_deviance_log - sm_deviance_log < 0.01, "expected h2o to have an equivalent or better deviance measures"

    print("Create h2o models with link: IDENTITY")
    h2o_model_id = h2o.glm(x=h2o_data[myX], y=h2o_data[myY], family="poisson", link="identity",alpha=[0.5], Lambda=[0])

    print("Create statsmodel models with link: IDENTITY")
    sm_model_id = sm.GLM(endog=sm_data_response, exog=sm_data_features,
                         family=sm.families.Poisson(sm.families.links.identity)).fit()

    print("Compare model deviances for link function identity")
    h2o_deviance_id = h2o_model_id.residual_deviance() / h2o_model_id.null_deviance()
    sm_deviance_id = sm_model_id.deviance / sm_model_id.null_deviance
    assert h2o_deviance_id - sm_deviance_id < 0.01, "expected h2o to have an equivalent or better deviance measures"

if __name__ == "__main__":
    h2o.run_test(sys.argv, link_functions_poisson)


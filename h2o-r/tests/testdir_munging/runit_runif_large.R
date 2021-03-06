setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source('../h2o-runit.R')

test.runif <- function(conn) {
    uploaded_frame <- h2o.uploadFile(conn, locate("bigdata/laptop/mnist/train.csv.gz"))
    r_u <- h2o.runif(uploaded_frame, seed=1234)

    imported_frame <- h2o.importFile(conn, locate("bigdata/laptop/mnist/train.csv.gz"))
    r_i <- h2o.runif(imported_frame, seed=1234)

    print(paste0("This demonstrates that seeding runif on identical frames with different chunk distributions ",
                 "provides different results. upload_file: ", mean(r_u), ", import_frame: ", mean(r_i)))

    testEnd()
}

doTest("Test runif", test.runif)

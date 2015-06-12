#FAQ


##Algorithms

**What does it mean if the r2 value in my model is negative?**

The coefficient of determination (also known as r^2) can be negative if: 

- linear regression is used without an intercept (constant)
- non-linear functions are fitted to the data
- predictions compared to the corresponding outcomes are not based on the model-fitting procedure using those data
- it is early in the build process (may self-correct as more trees are added)

If your r2 value is negative after your model is complete, your model is likely incorrect. Make sure your data is suitable for the type of model, then try adding an intercept. 

---

**How do I find the standard errors of the parameter estimates (p-values)?**

P-values are currently not supported. They are on our road map and will be added, depending on the current customer demand/priorities. Generally, adding p-values involves significant engineering effort because p-values for regularized GLM are not straightforward and have been defined only recently (with no standard implementation available that we know of). P-values for a restricted set of GLM problems (no regularization, low number of predictors) are easier to do and may be added sooner, if there is a sufficient demand.

For now, we recommend using a non-zero l1 penalty (alpha  > 0) and considering all non-zero coefficients in the model as significant. The recommended use case is running GLM with lambda search enabled and alpha > 0 and picking the best lambda value based on cross-validation or hold-out set validation.

---

##Clusters


**When trying to launch H2O, I received the following error message: `ERROR: Too many retries starting cloud.` What should I do?**

If you are trying to start a multi-node cluster where the nodes use multiple network interfaces, by default H2O will resort to using the default host (127.0.0.1). 

To specify an IP address, launch H2O using the following command: 

`java -jar h2o.jar -ip <IP_Address> -port <PortNumber>`

If this does not resolve the issue, try the following additional troubleshooting tips: 

- Test connectivity using `curl`: First, log in to the first node and enter `curl http://<Node2IP>:54321` (where `<Node2IP>` is the IP address of the second node. Then, log in to the second node and enter `curl http://<Node1IP>:54321` (where `<Node1IP>` is the IP address of the first node). Look for output from H2O. 
- Confirm ports 54321 and 54322 are available for both TCP and UDP. 
- Confirm your firewall is not preventing the nodes from locating each other. 
- Check if you have SELINUX or IPTABLES enabled; if so, disable them.  
- Check the configuration for the EC2 security group.
- Confirm that the username is the same on all nodes; if not, define the cloud using `-name`. 
- Check if the nodes are on different networks. 
- Check if the nodes have different interfaces; if so, use the `-network` option to define the network (for example, `-network 127.0.0.1`). 
- Force the bind address using `-ip`. 
- Confirm the nodes are not using different versions of H2O. 

---

**What should I do if I tried to start a cluster but the nodes started independent clouds that are not connected?**

Because the default cloud name is the user name of the node, if the nodes are on different operating systems (for example, one node is using Windows and the other uses OS X), the different user names on each machine will prevent the nodes from recognizing that they belong to the same cloud. To resolve this issue, use `-name` to configure the same name for all nodes. 

---

**One of the nodes in my cluster is unavailable — what do I do?**

H2O does not support high availability (HA). If a node in the cluster is unavailable, bring the cluster down and create a new healthy cluster. 

---

**How do I add new nodes to an existing cluster?**

New nodes can only be added if H2O has not started any jobs. Once H2O starts a task, it locks the cluster to prevent new nodes from joining. If H2O has started a job, you must create a new cluster to include additional nodes. 

---

**How do I check if all the nodes in the cluster are healthy and communicating?**

In the Flow web UI, click the **Admin** menu and select **Cluster Status**. 

---

**How do I create a cluster behind a firewall?**

H2O uses two ports: 

- The `REST_API` port (54321): Specify when launching H2O using `-port`; uses TCP only. 
- The `INTERNAL_COMMUNICATION` port (54322): Implied based on the port specified as the `REST_API` port, +1; requires TCP and UDP. 

You can start the cluster behind the firewall, but to reach it, you must make a tunnel to reach the `REST_API` port. To use the cluster, the `REST_API` port of at least one node must be reachable. 

---


**I launched H2O instances on my nodes - why won't they form a cloud?**

If you launch without specifying the IP address by adding argument -ip:

`$ java -Xmx20g -jar h2o.jar -flatfile flatfile.txt -port 54321`

and multiple local IP addresses are detected, H2O uses the default localhost (127.0.0.1) as shown below:

  ```
  10:26:32.266 main      WARN WATER: Multiple local IPs detected:
  +                                    /198.168.1.161  /198.168.58.102
  +                                  Attempting to determine correct address...
  10:26:32.284 main      WARN WATER: Failed to determine IP, falling back to localhost.
  10:26:32.325 main      INFO WATER: Internal communication uses port: 54322
  +                                  Listening for HTTP and REST traffic
  +                                  on http://127.0.0.1:54321/
  10:26:32.378 main      WARN WATER: Flatfile configuration does not include self:
  /127.0.0.1:54321 but contains [/192.168.1.161:54321, /192.168.1.162:54321]
  ```

To avoid using 127.0.0.1 on servers with multiple local IP addresses, run the command with the -ip argument to force H2O to launch at the specified IP:

`$ java -Xmx20g -jar h2o.jar -flatfile flatfile.txt -ip 192.168.1.161 -port 54321`

---

##Data

**How should I format my SVMLight data before importing?**

The data must be formatted as a sorted list of unique integers, the column indices must be >= 1, and the columns must be in ascending order. 

---

##General

**How do I score using an exported JSON model?**

Since JSON is just a representation format, it cannot be directly executed, so a JSON export can't be used for scoring. However, you can score by: 

- including the POJO in your execution stream and handing it observations one at a time 

  or

- handing your data in bulk to an H2O cluster, which will score using high throughput parallel and distributed bulk scoring. 


---

**How do I predict using multiple response variables?**

Currently, H2O does not support multiple response variables. To predict different response variables, build multiple modes. 

---

**How do I kill any running instances of H2O?**

In Terminal, enter `ps -efww | grep h2o`, then kill any running PIDs. You can also find the running instance in Terminal and press **Ctrl + C** on your keyboard. 

---


**Why is H2O not launching from the command line?**

	$ java -jar h2o.jar &
	% Exception in thread "main" java.lang.ExceptionInInitializerError
	at java.lang.Class.initializeClass(libgcj.so.10)
	at water.Boot.getMD5(Boot.java:73)
	at water.Boot.<init>(Boot.java:114)
	at water.Boot.<clinit>(Boot.java:57)
	at java.lang.Class.initializeClass(libgcj.so.10)
    Caused by: java.lang.IllegalArgumentException
    at java.util.regex.Pattern.compile(libgcj.so.10)
    at water.util.Utils.<clinit>(Utils.java:1286)
    at java.lang.Class.initializeClass(libgcj.so.10)
    ...4 more

The only prerequisite for running H2O is a compatible version of Java. We recommend Oracle's [Java 1.7](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html).

  
---

**Why did I receive the following error when I tried to launch H2O?**

```
[root@sandbox h2o-dev-0.3.0.1188-hdp2.2]hadoop jar h2odriver.jar -nodes 2 -mapperXmx 1g -output hdfsOutputDirName
Determining driver host interface for mapper->driver callback...
   [Possible callback IP address: 10.0.2.15]
   [Possible callback IP address: 127.0.0.1]
Using mapper->driver callback IP address and port: 10.0.2.15:41188
(You can override these with -driverif and -driverport.)
Memory Settings:
   mapreduce.map.java.opts:     -Xms1g -Xmx1g -Dlog4j.defaultInitOverride=true
   Extra memory percent:        10
   mapreduce.map.memory.mb:     1126
15/05/08 02:33:40 INFO impl.TimelineClientImpl: Timeline service address: http://sandbox.hortonworks.com:8188/ws/v1/timeline/
15/05/08 02:33:41 INFO client.RMProxy: Connecting to ResourceManager at sandbox.hortonworks.com/10.0.2.15:8050
15/05/08 02:33:47 INFO mapreduce.JobSubmitter: number of splits:2
15/05/08 02:33:48 INFO mapreduce.JobSubmitter: Submitting tokens for job: job_1431052132967_0001
15/05/08 02:33:51 INFO impl.YarnClientImpl: Submitted application application_1431052132967_0001
15/05/08 02:33:51 INFO mapreduce.Job: The url to track the job: http://sandbox.hortonworks.com:8088/proxy/application_1431052132967_0001/
Job name 'H2O_3889' submitted
JobTracker job ID is 'job_1431052132967_0001'
For YARN users, logs command is 'yarn logs -applicationId application_1431052132967_0001'
Waiting for H2O cluster to come up...
H2O node 10.0.2.15:54321 requested flatfile
ERROR: Timed out waiting for H2O cluster to come up (120 seconds)
ERROR: (Try specifying the -timeout option to increase the waiting time limit)
15/05/08 02:35:59 INFO impl.TimelineClientImpl: Timeline service address: http://sandbox.hortonworks.com:8188/ws/v1/timeline/
15/05/08 02:35:59 INFO client.RMProxy: Connecting to ResourceManager at sandbox.hortonworks.com/10.0.2.15:8050

----- YARN cluster metrics -----
Number of YARN worker nodes: 1

----- Nodes -----
Node: http://sandbox.hortonworks.com:8042 Rack: /default-rack, RUNNING, 1 containers used, 0.2 / 2.2 GB used, 1 / 8 vcores used

----- Queues -----
Queue name:            default
   Queue state:       RUNNING
   Current capacity:  0.11
   Capacity:          1.00
   Maximum capacity:  1.00
   Application count: 1
   ----- Applications in this queue -----
   Application ID:                  application_1431052132967_0001 (H2O_3889)
       Started:                     root (Fri May 08 02:33:50 UTC 2015)
       Application state:           FINISHED
       Tracking URL:                http://sandbox.hortonworks.com:8088/proxy/application_1431052132967_0001/jobhistory/job/job_1431052132967_0001
       Queue name:                  default
       Used/Reserved containers:    1 / 0
       Needed/Used/Reserved memory: 0.2 GB / 0.2 GB / 0.0 GB
       Needed/Used/Reserved vcores: 1 / 1 / 0

Queue 'default' approximate utilization: 0.2 / 2.2 GB used, 1 / 8 vcores used

----------------------------------------------------------------------

ERROR:   Job memory request (2.2 GB) exceeds available YARN cluster memory (2.2 GB)
WARNING: Job memory request (2.2 GB) exceeds queue available memory capacity (2.0 GB)
ERROR:   Only 1 out of the requested 2 worker containers were started due to YARN cluster resource limitations

----------------------------------------------------------------------
Attempting to clean up hadoop job...
15/05/08 02:35:59 INFO impl.YarnClientImpl: Killed application application_1431052132967_0001
Killed.
[root@sandbox h2o-dev-0.3.0.1188-hdp2.2]#
```

The H2O launch failed because more memory was requested than was available. Make sure you are not trying to specify more memory in the launch parameters than you have available. 

---
##Hadoop

<!---
>commenting out as in progress per Michal
**Why did I get an error in R when I tried to save my model to my home directory in Hadoop?**

To save the model in HDFS, prepend the save directory with `hdfs://`:

```
# build model
model = h2o.glm(model params)

# save model
hdfs_name_node <- "mr-0x6"
hdfs_tmp_dir <- "/tmp/runit”
model_path <- sprintf("hdfs://%s%s", hdfs_name_node, hdfs_tmp_dir)
h2o.saveModel(model, dir = model_path, name = “mymodel")
```

---
-->

**How do I specify which nodes should run H2O in a Hadoop cluster?**

Currently, this is not yet supported. To provide resource isolation (for example, to isolate H2O to the worker nodes, rather than the master nodes), use YARN Nodemanagers to specify the nodes to use. 

---

##Sparkling Water

**How do I inspect H2O using Flow while a droplet is running?**

If your droplet execution time is very short, add a simple sleep statement to your code: 

`Thread.sleep(...)`

---

**How do I change the memory size of the executors in a droplet?**

There are two ways to do this: 

- Change your default Spark setup in `$SPARK_HOME/conf/spark-defaults.conf`

  or 

- Pass `--conf` via spark-submit when you launch your droplet (e.g., `$SPARK_HOME/bin/spark-submit --conf spark.executor.memory=4g --master $MASTER --class org.my.Droplet $TOPDIR/assembly/build/libs/droplet.jar`

---

**I received the following error while running Sparkling Water using multiple nodes, but not when using a single node - what should I do?**

```
onExCompletion for water.parser.ParseDataset$MultiFileParseTask@31cd4150
water.DException$DistributedException: from /10.23.36.177:54321; by class water.parser.ParseDataset$MultiFileParseTask; class water.DException$DistributedException: from /10.23.36.177:54325; by class water.parser.ParseDataset$MultiFileParseTask; class water.DException$DistributedException: from /10.23.36.178:54325; by class water.parser.ParseDataset$MultiFileParseTask$DistributedParse; class java.lang.NullPointerException: null
	at water.persist.PersistManager.load(PersistManager.java:141)
	at water.Value.loadPersist(Value.java:226)
	at water.Value.memOrLoad(Value.java:123)
	at water.Value.get(Value.java:137)
	at water.fvec.Vec.chunkForChunkIdx(Vec.java:794)
	at water.fvec.ByteVec.chunkForChunkIdx(ByteVec.java:18)
	at water.fvec.ByteVec.chunkForChunkIdx(ByteVec.java:14)
	at water.MRTask.compute2(MRTask.java:426)
	at water.MRTask.compute2(MRTask.java:398)
```

This error output displays if the input file is not present on all nodes. Because of the way that Sparkling Water distributes data, the input file is required on all nodes (including remote), not just the primary node. Make sure there is a copy of the input file on all the nodes, then try again. 

---

##R

**How can I install the H2O R package if I am having permissions problems?**

This issue typically occurs for Linux users when the R software was installed by a root user. For more information, refer to the following [link](https://stat.ethz.ch/R-manual/R-devel/library/base/html/libPaths.html). 

To specify the installation location for the R packages, create a file that contains the `R_LIBS_USER` environment variable:

`echo R_LIBS_USER=\"~/.Rlibrary\" > ~/.Renviron`

Confirm the file was created successfully using `cat`: 

`$ cat ~/.Renviron`

You should see the following output:
 
`R_LIBS_USER="~/.Rlibrary"`

Create a new directory for the environment variable:

`$ mkdir ~/.Rlibrary`

Start R and enter the following: 

`.libPaths()`

Look for the following output to confirm the changes: 

```
[1] "<Your home directory>/.Rlibrary"                                         
[2] "/Library/Frameworks/R.framework/Versions/3.1/Resources/library"
```



---

##Tunneling between servers with H2O

To tunnel between servers (for example, due to firewalls): 

1. Use ssh to log in to the machine where H2O will run.
2. Start an instance of H2O by locating the working directory and calling a java command similar to the following example. 

 The port number chosen here is arbitrary; yours may be different.

 `$ java -jar h2o.jar -port  55599`

 This returns output similar to the following:

	```
	irene@mr-0x3:~/target$ java -jar h2o.jar -port 55599
	04:48:58.053 main      INFO WATER: ----- H2O started -----
	04:48:58.055 main      INFO WATER: Build git branch: master
	04:48:58.055 main      INFO WATER: Build git hash: 64fe68c59ced5875ac6bac26a784ce210ef9f7a0
	04:48:58.055 main      INFO WATER: Build git describe: 64fe68c
	04:48:58.055 main      INFO WATER: Build project version: 1.7.0.99999
	04:48:58.055 main      INFO WATER: Built by: 'Irene'
	04:48:58.055 main      INFO WATER: Built on: 'Wed Sep  4 07:30:45 PDT 2013'
	04:48:58.055 main      INFO WATER: Java availableProcessors: 4
	04:48:58.059 main      INFO WATER: Java heap totalMemory: 0.47 gb
	04:48:58.059 main      INFO WATER: Java heap maxMemory: 6.96 gb
	04:48:58.060 main      INFO WATER: ICE root: '/tmp'
	04:48:58.081 main      INFO WATER: Internal communication uses port: 55600
	+                                  Listening for HTTP and REST traffic on
	+                                  http://192.168.1.173:55599/
	04:48:58.109 main      INFO WATER: H2O cloud name: 'irene'
	04:48:58.109 main      INFO WATER: (v1.7.0.99999) 'irene' on
	/192.168.1.173:55599, discovery address /230 .252.255.19:59132
	04:48:58.111 main      INFO WATER: Cloud of size 1 formed [/192.168.1.173:55599]
	04:48:58.247 main      INFO WATER: Log dir: '/tmp/h2ologs'
	```

3. Log into the remote machine where the running instance of H2O will be forwarded using a command similar to the following (your specified port numbers and IP address will be different)

 	`ssh -L 55577:localhost:55599 irene@192.168.1.173`

4. Check the cluster status.

You are now using H2O from localhost:55577, but the
instance of H2O is running on the remote server (in this
case the server with the ip address 192.168.1.xxx) at port number 55599.

To see this in action note that the web UI is pointed at
localhost:55577, but that the cluster status shows the cluster running
on 192.168.1.173:55599

    
---


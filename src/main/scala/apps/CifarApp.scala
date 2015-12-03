package apps

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import libs._
import loaders._
import preprocessing._

// for this app to work, $SPARKNET_HOME should be the SparkNet root directory
// and you need to run $SPARKNET_HOME/caffe/data/cifar10/get_cifar10.sh
object CifarApp {
  val trainBatchSize = 100
  val testBatchSize = 100
  val channels = 3
  val width = 32
  val height = 32
  val imShape = Array(channels, height, width)
  val size = imShape.product

  // initialize nets on workers
  val sparkNetHome = "/home/zhunan/code/SparkNet"
  System.load(sparkNetHome + "/build/libccaffe.so")
  var netParameter = ProtoLoader.loadNetPrototxt(sparkNetHome +
    "/caffe/examples/cifar10/cifar10_full_train_test.prototxt")
  netParameter = ProtoLoader.replaceDataLayers(netParameter, trainBatchSize, testBatchSize, channels, height, width)
  val solverParameter = ProtoLoader.loadSolverPrototxtWithNet(sparkNetHome +
    "/caffe/examples/cifar10/cifar10_full_solver.prototxt", netParameter, None)
  val net = CaffeNet(solverParameter)

  def main(args: Array[String]) {
    val numWorkers = args(0).toInt
    val conf = new SparkConf()
      .setAppName("Cifar")
      .set("spark.driver.maxResultSize", "5G")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)

    // information for logging
    val startTime = System.currentTimeMillis()
    val trainingLog = new PrintWriter(new File("training_log_" + startTime.toString + ".txt" ))
    def log(message: String, i: Int = -1) {
      val elapsedTime = 1F * (System.currentTimeMillis() - startTime) / 1000
      if (i == -1) {
        trainingLog.write(elapsedTime.toString + ": "  + message + "\n")
      } else {
        trainingLog.write(elapsedTime.toString + ", i = " + i.toString + ": "+ message + "\n")
      }
      trainingLog.flush()
    }

    var netWeights = net.getWeights()

    val loader = new CifarLoader(sparkNetHome + "/caffe/data/cifar10/")
    log("loading train data")
    var trainRDD = sc.parallelize(loader.trainImages.zip(loader.trainLabels))
    log("loading test data")
    var testRDD = sc.parallelize(loader.testImages.zip(loader.testLabels))

    log("repartition data")
    trainRDD = trainRDD.repartition(numWorkers)
    testRDD = testRDD.repartition(numWorkers)

    log("processing train data")
    // we are training the mini batches in parallel
    val trainConverter = new ScaleAndConvert(trainBatchSize, height, width)
    val trainMiniBatchRDD = trainConverter.makeMiniBatchRDDWithoutCompression(trainRDD).persist()
    val numTrainMiniBatches = trainMiniBatchRDD.count()
    log("numTrainMinibatches = " + numTrainMiniBatches.toString)

    log("processing test data")
    val testConverter = new ScaleAndConvert(testBatchSize, height, width)
    val testMiniBatchRDD = testConverter.makeMiniBatchRDDWithoutCompression(testRDD).persist()
    val numTestMiniBatches = testMiniBatchRDD.count()
    log("numTestMinibatches = " + numTestMiniBatches.toString)

    // total number of data entries
    val numTrainData = numTrainMiniBatches * trainBatchSize
    val numTestData = numTestMiniBatches * testBatchSize

    val trainPartitionSizes = trainMiniBatchRDD.mapPartitions(iter => Array(iter.size).iterator).persist()
    val testPartitionSizes = testMiniBatchRDD.mapPartitions(iter => Array(iter.size).iterator).persist()
    log("trainPartitionSizes = " + trainPartitionSizes.collect().deep.toString)
    log("testPartitionSizes = " + testPartitionSizes.collect().deep.toString)

    //how many partitions we have, for each partition we train an independent net
    val partitions = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    var i = 0
    while (true) {
      log("broadcasting weights", i)
      val broadcastWeights = sc.broadcast(netWeights)
      log("setting weights on workers", i)
      partitions.foreach(_ => net.setWeights(broadcastWeights.value))

      if (i % 10 == 0) {
        log("testing, i")
        val testScores = testPartitionSizes.zipPartitions(testMiniBatchRDD) (
          (lenIt, testMinibatchIt) => {
            assert(lenIt.hasNext && testMinibatchIt.hasNext)
            val len = lenIt.next
            assert(!lenIt.hasNext)
            val minibatchSampler = new MiniBatchSampler(testMinibatchIt, len, len)
            net.setTestData(minibatchSampler, len, None)
            Array(net.test()).iterator // do testing
          }
        ).cache()
        val testScoresAggregate = testScores.reduce((a, b) => (a, b).zipped.map(_ + _))
        val accuracies = testScoresAggregate.map(v => 100F * v / numTestMiniBatches)
        log("%.2f".format(accuracies(0)) + "% accuracy", i)
      }

      log("training", i)
      val syncInterval = 10

      trainPartitionSizes.zipPartitions(trainMiniBatchRDD) (
        (lenIt, trainMinibatchIt) => {
          assert(lenIt.hasNext && trainMinibatchIt.hasNext)
          val len = lenIt.next
          assert(!lenIt.hasNext)
          val miniBatchSampler = new MiniBatchSampler(trainMinibatchIt, len, syncInterval)
          net.setTrainData(miniBatchSampler, None)
          //we synchronize the parameter for every syncInterval iterations
          net.train(syncInterval)
          Array(0).iterator
        }
      ).foreachPartition(_ => ()) // trigger a job

      log("collecting weights", i)
      netWeights = partitions.map(_ => { net.getWeights() }).reduce((a, b) => WeightCollection.add(a, b))
      netWeights.scalarDivide(1F * numWorkers)
      i += 1
    }

    log("finished training")
  }
}

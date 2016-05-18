

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object PrepareData {

  def main(args: Array[String]) {

    val sparkConf = new SparkConf().setAppName("PrepareData").setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val sqlc = new SQLContext(sc)

    import sqlc.implicits._

    val repartition = 4
    val dataDir = "/Users/kobets/beeline_data/"
    val pairsGraph_path = dataDir + "train_new.csv"
    val companies_path = dataDir + "companies.csv"
    val personsRaw_path = dataDir + "persons"
    val friendsListGraph_path = dataDir + "friendsListGraph"



    var files = sc.textFile(pairsGraph_path)
      .map(line => {
        val lineSplit = line.split(",")
        val person = lineSplit(0).toInt
        val interaction = lineSplit(8).toInt
        person -> interaction
      })

    val files_B = sc.textFile(pairsGraph_path)
      .map(line => {
        val lineSplit = line.split(",")
        val person = lineSplit(1).toInt
        val interaction = lineSplit(9).toInt
        person -> interaction
      })

    files = files ++ files_B

    files = files
      .reduceByKey((x, y) => x + y)


    var calls = sc.textFile(pairsGraph_path)
      .map(line => {
        val lineSplit = line.split(",")
        val person = lineSplit(0).toInt
        val interaction = lineSplit(6).toInt
        person -> interaction
      })

    val calls_B = sc.textFile(pairsGraph_path)
      .map(line => {
        val lineSplit = line.split(",")
        val person = lineSplit(1).toInt
        val interaction = lineSplit(7).toInt
        person -> interaction
      })

    calls = calls ++ calls_B

    calls = calls
      .reduceByKey((x, y) => x + y)


    var messages = sc.textFile(pairsGraph_path)
      .map(line => {
        val lineSplit = line.split(",")
        val person = lineSplit(0).toInt
        val interaction = lineSplit(4).toInt
        person -> interaction
      })

    val messages_B = sc.textFile(pairsGraph_path)
      .map(line => {
        val lineSplit = line.split(",")
        val person = lineSplit(1).toInt
        val interaction = lineSplit(5).toInt
        person -> interaction
      })

    messages = messages ++ messages_B

    messages = messages
      .reduceByKey((x, y) => x + y)


    val companies = sc.textFile(companies_path)
      .map(line => {
        val lineSplit = line.split(",")
        lineSplit(0).toInt -> lineSplit(1).toInt
      })


    var graph_pairs = sc.textFile(pairsGraph_path)
      .map(line => {
        val lineSplit = line.split(",")
        val userA = lineSplit(0).toInt
        val userB = lineSplit(1).toInt
        (userA, userB)
      })

    val graph_pairs_reversed = graph_pairs
      .map(t => (t._2, t._1))

    graph_pairs = graph_pairs ++ graph_pairs_reversed



    val graph = graph_pairs
      .groupByKey
      .map(t => t._1 -> t._2.toArray.sorted)

    graph.toDF.repartition(repartition).write.parquet(friendsListGraph_path)

    val friends = sqlc.read.parquet(friendsListGraph_path)
      .map(t => t.getAs[Int](0) -> t.getAs[Seq[Int]](1).length)


    companies
      .leftOuterJoin(messages)
      .leftOuterJoin(calls)
      .leftOuterJoin(files)
      .leftOuterJoin(friends)
      .sortByKey(ascending = true, 1)
      .map(t => t._1 + "," + t._2._1._1._1._1 + "," + t._2._1._1._1._2.get + "," + t._2._1._1._2.get + "," + t._2._1._2.get + "," + t._2._2.get)
      .repartition(1)
      .saveAsTextFile(personsRaw_path)  // rename to persons.csv and move to root of dataDir

  }
}
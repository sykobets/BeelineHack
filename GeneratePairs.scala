
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer


case class PairWithCommonFriends(person1: Int, person2: Int, commonFriendsCount: Double)

object GeneratePairs {

  def main(args: Array[String]) {

    val sparkConf = new SparkConf().setAppName("BeelineHack").setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val sqlc = new SQLContext(sc)

    import sqlc.implicits._


    val numPartitionsGraph = 100
    val repartition = 4
    val dataDir = "/Users/kobets/beeline_data/"
    val friendsListGraph_path = dataDir + "friendsListGraph"
    val commonFriendsCounts_path = dataDir + "commonFriendsCounts"
    val AdamicAdarCoefs_path = dataDir + "AdamicAdarCoefs"


    val friends = sqlc.read.parquet(friendsListGraph_path)
      .map(t => t.getAs[Int](0) -> t.getAs[Seq[Int]](1).length)


    def generatePairs_for_commonFriendsCounts(pplWithCommonFriends: Seq[Int], numPartitions: Int, k: Int) = {
      val pairs = ArrayBuffer.empty[((Int, Int), Double)]
      for (i <- 0 until pplWithCommonFriends.length) {
        if (pplWithCommonFriends(i) % numPartitions == k) {
          for (j <- i + 1 until pplWithCommonFriends.length) {
            pairs.append((pplWithCommonFriends(i), pplWithCommonFriends(j)) -> 1)
          }
        }
      }
      pairs
    }

    def generatePairs_for_AdamicAdarCoefs(pplWithCommonFriends: Seq[Int], numPartitions: Int, k: Int) = {
      val pairs = ArrayBuffer.empty[((Int, Int), Double)]
      for (i <- 0 until pplWithCommonFriends.length) {
        if (pplWithCommonFriends(i) % numPartitions == k) {
          for (j <- i + 1 until pplWithCommonFriends.length) {
            pairs.append((pplWithCommonFriends(i), pplWithCommonFriends(j)) -> 1/math.log(pplWithCommonFriends.length))
          }
        }
      }
      pairs
    }

    for (k <- 0 until numPartitionsGraph) {
      val commonFriendsCounts = sqlc.read.parquet(friendsListGraph_path)
        .map(t => t.getAs[Seq[Int]](1))
        .filter(t => t.length >= 2)
        .map(t => generatePairs_for_commonFriendsCounts(t, numPartitionsGraph, k))
        .flatMap(t => t)
        .reduceByKey((x, y) => x + y)
        .map(t => PairWithCommonFriends(t._1._1, t._1._2, t._2))

      commonFriendsCounts.toDF.repartition(repartition).write.parquet(commonFriendsCounts_path + "/part_" + k)


      val AdamicAdarCoefs = sqlc.read.parquet(friendsListGraph_path)
        .map(t => t.getAs[Seq[Int]](1))
        .filter(t => t.length >= 2)
        .map(t => generatePairs_for_AdamicAdarCoefs(t, numPartitionsGraph, k))
        .flatMap(t => t)
        .reduceByKey((x, y) => x + y)
        .map(t => PairWithCommonFriends(t._1._1, t._1._2, t._2))

      AdamicAdarCoefs.toDF.repartition(repartition).write.parquet(AdamicAdarCoefs_path + "/part_" + k)

    }


  }

}

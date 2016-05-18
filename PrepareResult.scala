

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object PrepareResult {

  def main(args: Array[String]) {

    val sparkConf = new SparkConf().setAppName("BeelineHack").setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val sqlc = new SQLContext(sc)

    val dataDir = "/Users/kobets/beeline_data/"
    val predictionRaw_path = dataDir + "predictionRaw"
    val prediction_path = dataDir + "prediction"


    val prediction = sqlc.read.parquet(predictionRaw_path)
      .map(t => t.getInt(0) -> t.getStruct(1))
      .map(t => t._1 ->(t._2.getInt(0), t._2.getDouble(1)))


    prediction
      .sortBy(-_._2._2)
      .map(t => t._1 -> t._2._1)
      .filterByRange(1, 20000000)
      .groupByKey(200)
      .map(t => t._1 + "," + t._2.mkString(","))
      .repartition(1).saveAsTextFile(prediction_path)
  }

}

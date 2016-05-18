import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by kobets on 18.05.16.
  */
object Prediction {


  def main(args: Array[String]) {

    val sparkConf = new SparkConf().setAppName("BeelineHack").setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val sqlc = new SQLContext(sc)

    import sqlc.implicits._


    val dataDir = "/Users/kobets/beeline_data/"
    val pairsGraph_path = dataDir + "train_new.csv"
    val companiesLinks_path = dataDir + "companies_links.csv"
    val persons_path = dataDir + "persons.csv"
    val commonFriendsCounts_path = dataDir + "commonFriendsCounts"
    val AdamicAdarCoefs_path = dataDir + "AdamicAdarCoefs"
    val model_path = dataDir + "LogisticRegressionModel"
    val predictionRaw_path = dataDir + "predictionRaw"



    val positives = sc.textFile(pairsGraph_path)
      .map(line => {
        val lineSplit = line.split(",")
        val person_A = lineSplit(0).toInt
        val person_B = lineSplit(1).toInt
        (person_A, person_B) -> 1.0
      })


    val personsInfo = {
      sc.textFile(persons_path)
        .map(line => {
          val lineSplit = line.trim().split(",")
          val company_id = lineSplit(1).toInt
          val company_dumm = Array(0, 0, 0, 0, 0)
          company_dumm(company_id) = 1
          lineSplit(0).toInt -> PersonInfo(company_id, company_dumm, lineSplit(2).toInt, lineSplit(3).toInt, lineSplit(4).toInt, lineSplit(5).toInt)
        })
    }

    val personsInfoBC = sc.broadcast(personsInfo.collectAsMap())


    val companiesLinks = {
      sc.textFile(companiesLinks_path)
        .map(line => {
          val lineSplit = line.trim().split(",")
          (lineSplit(0).toInt, lineSplit(1).toInt) -> CompaniesInfo(lineSplit(2).toInt, lineSplit(3).toLong)
        })
    }

    val companiesLinksBC = sc.broadcast(companiesLinks.collectAsMap())



    def prepareScalarData(
                           pairsInfo: RDD[((Int, Int), PairInfo)],
                           personsInfoBC: Broadcast[scala.collection.Map[Int, PersonInfo]],
                           companiesLinksBC: Broadcast[scala.collection.Map[(Int, Int), CompaniesInfo]],
                           positives: RDD[((Int, Int), Double)]) = {
      pairsInfo
        .map(pair => {
          val person_A = pair._1._1
          val person_B = pair._1._2

          val company_A = personsInfoBC.value.get(person_A).get.company_id
          val company_B = personsInfoBC.value.get(person_B).get.company_id
          val companies_pair = (company_A.min(company_B), company_A.max(company_B))
          val companies_links = companiesLinksBC.value.get(companies_pair).get.links
          val companies_potential = companiesLinksBC.value.get(companies_pair).get.potential
          val companies_adamic_adar_coef = companies_links / (companies_potential - companies_links)

          val friends_A = personsInfoBC.value.get(person_A).get.friends
          val friends_B = personsInfoBC.value.get(person_B).get.friends

          val common_friends_count = pair._2.common_friends_count
          val adamic_adar_coef = pair._2.adamic_adar_coef
          val jaccard_coef = common_friends_count / (friends_A + friends_B - common_friends_count)
          val random_coef = adamic_adar_coef * (1 / friends_A + 1 / friends_B) / 2

          (person_A, person_B) -> Vectors.dense(
            friends_A,
            friends_B,
            friends_A * friends_B,
            common_friends_count,
            adamic_adar_coef,
            jaccard_coef,
            random_coef,
            companies_links,
            companies_potential,
            companies_adamic_adar_coef,
            personsInfoBC.value.get(person_A).get.messages,
            personsInfoBC.value.get(person_B).get.messages,
            personsInfoBC.value.get(person_A).get.calls,
            personsInfoBC.value.get(person_B).get.calls,
            personsInfoBC.value.get(person_A).get.files,
            personsInfoBC.value.get(person_B).get.files)
        })
        .leftOuterJoin(positives)
    }

    def prepareDummieData(
                           pairsInfo: RDD[((Int, Int), PairInfo)],
                           personsInfoBC: Broadcast[scala.collection.Map[Int, PersonInfo]],
                           positives: RDD[((Int, Int), Double)]) = {
      pairsInfo
        .map(pair => {
          val person_A = pair._1._1
          val person_B = pair._1._2

          (person_A, person_B) -> Vectors.dense(
            personsInfoBC.value.get(person_A).get.company_dumm(0),
            personsInfoBC.value.get(person_A).get.company_dumm(1),
            personsInfoBC.value.get(person_A).get.company_dumm(2),
            personsInfoBC.value.get(person_A).get.company_dumm(3),
            personsInfoBC.value.get(person_A).get.company_dumm(4),
            personsInfoBC.value.get(person_B).get.company_dumm(0),
            personsInfoBC.value.get(person_B).get.company_dumm(1),
            personsInfoBC.value.get(person_B).get.company_dumm(2),
            personsInfoBC.value.get(person_B).get.company_dumm(3),
            personsInfoBC.value.get(person_B).get.company_dumm(4))
        })
        .leftOuterJoin(positives)
    }




    val test_commonFriendsCounts = sqlc.read.parquet(commonFriendsCounts_path + "/part_*")
      .map(t => (t.getAs[Int](0), t.getAs[Int](1)) -> t.getAs[Double](2))


    val test_AdamicAdarCoefs = sqlc.read.parquet(AdamicAdarCoefs_path + "/part_*")
      .map(t => (t.getAs[Int](0), t.getAs[Int](1)) -> t.getAs[Double](2))


    val test_pairsInfo = test_commonFriendsCounts
      .leftOuterJoin(test_AdamicAdarCoefs)
      .map(t => (t._1._1, t._1._2) -> PairInfo(t._2._1, t._2._2.get))


    val test_data_scalar = prepareScalarData(test_pairsInfo, personsInfoBC, companiesLinksBC, positives)
      .map(t => t._1 -> LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
      .filter(t => t._2.label == 0.0)


    val test_scaler = new StandardScaler(withMean = true, withStd = true).fit(test_data_scalar.map(x => x._2.features))

    val test_data_scalar_scaled = test_data_scalar.map(x => x._1 -> LabeledPoint(x._2.label, test_scaler.transform(Vectors.dense(x._2.features.toArray))))



    val test_data_dummie = prepareDummieData(test_pairsInfo, personsInfoBC, positives)
      .map(t => t._1 -> LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
      .filter(t => t._2.label == 0.0)


    val test_data = test_data_scalar_scaled
      .leftOuterJoin(test_data_dummie)
      .map(t => {
        val features = t._2._1.features.toArray ++ t._2._2.get.features.toArray
        t._1 -> LabeledPoint(t._2._1.label, Vectors.dense(features))
      })



    val model = LogisticRegressionModel.load(sc, model_path)

    val prediction = test_data
      .flatMap { case (id, LabeledPoint(label, features)) =>
        val prediction = model.predict(features)
        Seq(id._1 -> (id._2, prediction), id._2 -> (id._1, prediction))
      }

    prediction.repartition(100).toDF().write.parquet(predictionRaw_path)



  }

}

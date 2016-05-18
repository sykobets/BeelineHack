import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


case class CompaniesInfo(links: Int,
                         potential: Long)

case class PairInfo(common_friends_count: Double,
                    adamic_adar_coef: Double)

case class PersonInfo(company_id: Int,
                      company_dumm: Array[Int],
                      messages: Int,
                      calls: Int,
                      files: Int,
                      friends: Int)

object Training {

  def main(args: Array[String]) {

    val sparkConf = new SparkConf().setAppName("BeelineHack").setMaster("local[2]")
    val sc = new SparkContext(sparkConf)
    val sqlc = new SQLContext(sc)

    val dataDir = "/Users/kobets/beeline_data/"
    val pairsGraph_path = dataDir + "train_new.csv"
    val companiesLinks_path = dataDir + "companies_links.csv"
    val persons_path = dataDir + "persons.csv"
    val commonFriendsCounts_path = dataDir + "commonFriendsCounts"
    val AdamicAdarCoefs_path = dataDir + "AdamicAdarCoefs"
    val model_path = dataDir + "LogisticRegressionModel"



    val commonFriendsCounts = {
      sqlc.read.parquet(commonFriendsCounts_path + "/part_6[67]")
        .map(t => (t.getAs[Int](0), t.getAs[Int](1)) -> t.getAs[Double](2))
    }


    val AdamicAdarCoefs = {
      sqlc.read.parquet(AdamicAdarCoefs_path + "/part_6[67]")
        .map(t => (t.getAs[Int](0), t.getAs[Int](1)) -> t.getAs[Double](2))
    }


    val pairsInfo = commonFriendsCounts
      .leftOuterJoin(AdamicAdarCoefs)
      .map(t => (t._1._1, t._1._2) -> PairInfo(t._2._1, t._2._2.get))



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



    val data_scalar = {
      prepareScalarData(pairsInfo, personsInfoBC, companiesLinksBC, positives)
        .map(t => t._1 -> LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
    }

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(data_scalar.map(x => x._2.features))

    val data_scalar_scaled = data_scalar.map(x => x._1 -> LabeledPoint(x._2.label, scaler.transform(Vectors.dense(x._2.features.toArray))))


    val data_dummie = {
      prepareDummieData(pairsInfo, personsInfoBC, positives)
        .map(t => t._1 -> LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
    }


    val data = data_scalar_scaled
      .leftOuterJoin(data_dummie)
      .map(t => {
        val features = t._2._1.features.toArray ++ t._2._2.get.features.toArray
        LabeledPoint(t._2._1.label, Vectors.dense(features))
      })


    val splits = data.randomSplit(Array(0.2, 0.8), seed = 666L)
    val training = splits(0).cache()
    val validation = splits(1)



    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    model.clearThreshold()
    model.save(sc, model_path)

    val predictionAndLabels = {
      validation.map { case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
      }
    }


    @transient val metricsLogReg = new BinaryClassificationMetrics(predictionAndLabels, 100)

    val f1_score = metricsLogReg.fMeasureByThreshold(1.0)
    val roc_auc = metricsLogReg.areaUnderROC()

    println("F1_SCORE = " + f1_score.sortBy(-_._2).take(1)(0))
    println("ROC_AUC = " + roc_auc)

  }

}

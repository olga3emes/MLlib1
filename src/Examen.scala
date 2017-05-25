import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils

/**
  * Created by olga on 25/5/17.
  */
object Examen {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("Examen").setMaster("local")
    val sc = new SparkContext(conf)
    System.setProperty("hadoop.home.dir", "c:\\Winutil\\")

    //Load data
    val data =MLUtils.loadLabeledPoints(sc,"machinedata_mlutils.txt")

    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    //Decision Tree Regression -------------------------------------------------------------------------------------

    val numClasses = 2
    val maxDepth = 3 //profundidad del árbol
    val maxBins = 100
    val impurity = "entropy" //2 opciones: “gini” or “entropy”.

    val categoricalFeaturesInfo = Map[Int,Int]()
    val	impurityR= "variance"

    val modelR = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurityR, maxDepth, maxBins)

    val labelAndPredsR = testData.map { point =>
      val predictionR = modelR.predict(point.features)
      (point.label, predictionR)
    }

    labelAndPredsR.foreach((result2) => println(s"actual label: ${result2._1},predicted label : ${result2._2}"))

    val testMSE = labelAndPredsR.map{ case(v, p) => math.pow((v - p), 2)}.mean()

    println("Test Mean Squared Error = " + testMSE)
    println("Learned classification tree model:\n" + modelR.toDebugString)



  }

}

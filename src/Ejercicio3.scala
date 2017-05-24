import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.DecisionTree

/**
  * Created by olga on 24/5/17.
  */

/*Problema 3: Regresión
Una empresa dedicada a la fabricación de ordenadores está interesada en predecir el rendimiento de la CPU
de un ordenador apartir de las siguientes características hardware:
tiempo de un ciclo de instrucción en nanosegundos,
mínimo de memoria principal en kilobytes,
máximo de memoria principal en kilobytes,
memoria cache en kilobytes,
número mínimo de canales en unidades
número máximo de canales en unidades.
Para ello, se dispone del conjunto de datos machinedata.txt que contiene las características hardware descritas
anteriormente y el rendimiento de la CPU de 209 ordenadores.
Concretamente, cada instancia describe un ordenador en el que el primer valor corresponde al rendimiento de la CPU
 y los 6 valores siguientes corresponden a las características hardware. Se pide:

§Implementar una regresión lineal que prediga el rendimiento de los ordenadores de un conjunto de test. Tenga en cuenta
que las características hardware son variables que se encuentran en rangos muy dispares, por lo que es necesario
escalar dichos datos de forma que tengan media cero y desviación estándar uno.

§ Modifique el programa del apartado anterior para resolver el problema usando regularización con el objeto de evitar
overfitting.

§ Resuelva el mismo problema de regresión usando un árbol de decisión y compare los errores obtenidos por ambos
  métodos.*/


object Ejercicio3 {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN);
    Logger.getLogger("akka").setLevel(Level.WARN);
    //System.setProperty("hadoop.home.dir", "C:\\Winutil\\")

    val conf = new SparkConf().setAppName("Ejercicio3").setMaster("local")
    val sc = new SparkContext(conf)

    //Carga de datos

    val data =MLUtils.loadLabeledPoints(sc,"machinedata_mlutils.txt")

    /**Transformamos los datos para ajustar el modelo en las caracteristicas hardware***/


    //Discretizarlos

    val discretizedData = data.map { lp =>
      LabeledPoint(lp.label, Vectors.dense(lp.features.toArray))
    }
    //discretizedData.saveAsTextFile("reformat")

    //Escalar datos

    val standarScaler = new StandardScaler(withMean = false, withStd = true).fit(data.map(x => x.features))
    val scaled = data.map(x => (x.label, standarScaler.transform(Vectors.dense(x.features.toArray))))

    //scaled.saveAsTextFile("scaled")

    val data2 =MLUtils.loadLabeledPoints(sc,"scaled").cache()
    //val data2 =MLUtils.loadLabeledPoints(sc,"reformat").cache()

    val splits = data2.randomSplit(Array(0.7,0.3))

    val (trainingData, testData) = (splits(0),splits(1))

    val algorithm = new LinearRegressionWithSGD()
    algorithm.setIntercept(true)
    val numIterations=5
    val stepSize=1

    algorithm.optimizer.setNumIterations(numIterations)
    algorithm.optimizer.setStepSize(stepSize)
    val model= algorithm.run(data2)

    val valuesAndPreds = testData.map{ point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    valuesAndPreds.foreach((result) => println(s"actual label: ${result._1},predicted label : ${result._2}"))
    // println (valuesAndPreds)

    val MSE = valuesAndPreds.map{ case(v, p) =>
      math.pow((v-p),2)}.mean()


    println("Error: "+MSE)

    //Decision Tree Regression -------------------------------------------------------------------------------------

    val numClasses = 2
    val maxDepth = 3 //profundidad del árbol
    val maxBins = 100
    val impurity = "entropy" //2 opciones: “gini” or “entropy”.

    val categoricalFeaturesInfo = Map(0 -> 10)
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

/**
  * Created by olga on 24/5/17.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.mllib.util.MLUtils

/*Problema 1: Clasificación
 Una empresa tiene un big data con información de sus clientes y desea explotarlos para poder hacer ofertas personalizadas.
 Los datos contienen las siguientes columnas:
 tipo de cliente (variable binaria),
 tipo de producto contratado(variable discreta con 86 valores),
 tipo de mercado (variable binaria)
 consumo horario (25 variables reales).
   Usar para la evaluación un conjunto de training formado por el 70% de los datos y un conjunto de test formado por el 30%.
   Se pide:
 § Predecir el tipo de cliente a través de un árbol de decisión de profundidad máxima tres.

   https://spark.apache.org/docs/2.1.0/mllib-decision-tree.html

 § Predecir el tipo de cliente a través de un ensemble de árboles de decisión formado por cuatro árboles de profundidad
   máxima tres.

   https://spark.apache.org/docs/2.1.0/mllib-ensembles.html

 § En ambos casos, experimente con diferentes valores de la medida de impuridad y del parámetro maxBins rellenando la
 tabla que se muestre a continuación. ¿Con qué método de machine learning se obtienen los mejores resultados?*/

//DECISION TREE CLASSIFICATION
//maxBins=86  entropy error -> 14.873723089209065 gini error -> 14.91412372853093
//maxBins=100 entropy error -> 15.313031586671558 gini error -> 15.004666044527395
//maxBins=110 entropy error -> 15.023770737059078 gini error -> 15.068309670506293
//maxBins=120 entropy error -> 15.339086730813742 gini error -> 15.225469518931407

//RANDOM FOREST CLASSIFICATION
//maxBins=86  entropy error -> 14.900342727847468 gini error -> 14.817408704352175
//maxBins=100 entropy error -> 15.059537707214568 gini error -> 14.94800693240901
//maxBins=110 entropy error -> 15.116858938129592 gini error -> 14.817171175997856
//maxBins=120 entropy error -> 14.919745052891514 gini error -> 15.108299019115531



object Ejercicio1 {

  def main(args:	Array[String]):	Unit	= {
    Logger.getLogger("org").setLevel(Level.WARN);
    Logger.getLogger("akka").setLevel(Level.WARN);
    System.setProperty("hadoop.home.dir", "C:\\Winutil\\")

    val conf = new SparkConf().setAppName("Ejercicio1").setMaster("local")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLabeledPoints(sc, "Endesa_classCNAE_nc2_natr27.txt")

    val splits = data.randomSplit(Array(0.7, 0.3)) // conjunto de training 70%, conjunto de test 30%
    val (trainingData, testData) = (splits(0), splits(1))

    val numClasses = 2 //0 o 1 -> cliente es binario
    val maxDepth = 3 //profundidad del árbol
    val maxBins = 120 //number of bins used when discretizing continuous features (32 por defecto, necesitamos como mínimo 86)
    val impurity = "entropy" //2 opciones: “gini” or “entropy”.

    val categoricalFeaturesInfo = Map(0 -> 86, 1 -> 2) // 0 - tipo de producto(86 posibilidades) y  1- tipo de mercado (2 posibilidades)


    //Decision Tree Classification -------------------------------------------------------------------------------------

    println("Decision Tree Classification:\n")
    val model = DecisionTree.trainClassifier(trainingData, numClasses,
      categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val testError = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()*100 //(en %)
    println("Test Error Decision Tree = " + testError+"\n")
    println("Learned classification tree model:\n" + model.toDebugString)


    //Decision Tree Regression -------------------------------------------------------------------------------------

    /*val	impurityR= "variance"

    val modelR = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurityR, maxDepth, maxBins)

    val labelAndPredsR = testData.map { point =>
      val predictionR = modelR.predict(point.features)
      (point.label, predictionR)
    }

    val testMSE = labelAndPredsR.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    println("Test Mean Squared Error = " + testMSE)
    println("Learned classification tree model:\n" + modelR.toDebugString)

    //---------------------------------------*/

    val numTrees = 4
    val featureSubsetStrategy = "auto" // Let the algorithm choose. Valores posibles -> auto, all, sqrt, log2, onethird


    //Random Forest Classification -------------------------------------------------------------------------------------

    println("Random Forest Classification:\n")

    val randomForest = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)


    val labelAndPredsRF = testData.map { point =>
      val predictionRF = randomForest.predict(point.features)
      (point.label, predictionRF)
    }

    val testErrorRF = labelAndPredsRF.filter(r => r._1 != r._2).count().toDouble / testData.count()*100 // (en %)
    println("Test Error Random Forest = " + testErrorRF +"\n")
    println("Learned classification forest model:\n" + randomForest.toDebugString)

    //Random Forest Regression -----------------------------------------------------------------------------------------

   /* println("Random Forest Regression:\n")

    val seed=2

    val randomForestR = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed)

    val labelAndPredsRFR = testData.map { point =>
      val predictionRFR = randomForestR.predict(point.features)
      (point.label, predictionRFR)
    }

    val testRFR = labelAndPredsRFR.map{ case (v, p) => math.pow(v - p, 2) }.mean()
    println("Test Mean Squared Error = " + testRFR)
    println("Learned regression forest model:\n" + randomForestR.toDebugString) */



  }
}

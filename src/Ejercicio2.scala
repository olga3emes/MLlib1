/**
  * Created by olga on 24/5/17.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

/* Problema 2: Clustering
La misma empresa desea obtener patrones de comportamiento de sus clientes a partir de sus consumos horarios.
Se pide:
§ Obtener un clustering con dos clusters y analice a posteriori si en cada cluster se agrupan los clientes
 del mismo tipo.

 //Sí

 Within Set Sum of Squared Errors = 2.940433375185971E11

Labels: MapPartitionsRDD[105] at map at KMeansModel.scala:69

Final Centers:

[180.69066039644014,142.11992819579288,123.04142394822007,114.01143305016181,110.28115392394822,111.36223199838187,121.83264057443365,149.84364381067962,175.06786003236246,193.45235133495146,211.66879045307445,215.31261124595468,221.93528519417475,242.3046824433657,248.09345165857604,228.95021743527508,217.99919093851133,223.46654530744337,233.73873887540452,247.62743729773464,275.4353408171521,309.3987661812298,287.83831917475726,235.76931128640777,0.0]

[2670.5574866310158,2489.233065953654,2397.681818181818,2351.9237967914437,2327.194295900178,2348.3074866310158,2391.016934046346,2024.353386809269,2077.4647950089125,2460.180926916221,2781.3377896613188,2842.9282531194294,2891.701871657754,2851.202762923351,2535.5681818181815,2361.0726381461673,2337.5561497326203,2514.475490196078,2636.382798573975,2766.9532085561495,3059.8663101604275,3397.017825311943,3254.823975044563,2946.285204991087,0.0]


§ Modificar el programa anterior para obtener un clustering con ochenta y seis clusters y analice a
posteriori si en cada cluster se agrupan los productos contratados del mismo tipo.*/




//https://spark.apache.org/docs/2.1.0/mllib-clustering.html


object Ejercicio2 {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN);
    Logger.getLogger("akka").setLevel(Level.WARN);
    System.setProperty("hadoop.home.dir", "C:\\Winutil\\")

    val conf = new SparkConf().setAppName("Ejercicio2").setMaster("local")
    val sc = new SparkContext(conf)

    //Carga de datos,
    val data = sc.textFile("25consumos_clustering.txt")

    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // KMeans----------------------------------------------------------------------------------------------------------
    val numClusters = 86//2
    val numIterations = 20

    val inizializationMode ="random" //"random" or "k-means||" (default).

    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    //val clusters= KMeans.train(parsedData,numClusters,numIterations,inizializationMode)


    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE+"\n") //Dentro de la suma establecida de errores cuadrados

    //To obtain labels of the clusters
    val	labels	=	clusters.predict(parsedData)
    println("Labels: " +labels+"\n")

    //To obtain specified cluster's centroids
    // val	vect	=	clusters.clusterCenters(1) //2 clusters 0 y 1
    // println("Centroids: "+ vect +"\n")

    println("Final Centroids: ")
    clusters.clusterCenters.foreach(println)

   //-------------------------------------------------------------------------------------------------------------------

  }
}

/**
  * Created by olga on 24/5/17.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils


/*Problema 2: Clustering
La misma empresa desea obtener patrones de comportamiento de sus clientes a partir de sus consumos horarios. Se pide:
§ Obtener un clustering con dos clusters y analice a posteriori si en cada cluster se agrupan los clientes del mismo tipo.
§ Modificar el programa anterior para obtener un clustering con ochenta y seis clusters y analice a posteriori si en cada cluster se agrupan los productos contratados del mismo tipo.*/

object Ejercicio2 {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN);
    Logger.getLogger("akka").setLevel(Level.WARN);
    System.setProperty("hadoop.home.dir", "C:\\Winutil\\")

    val conf = new SparkConf().setAppName("Ejercicio2").setMaster("local")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLabeledPoints(sc, "25consumos_clustering.txt")






  }
}

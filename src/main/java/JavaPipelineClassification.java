import org.apache.spark.ml.Model;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;
import scala.collection.mutable.WrappedArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by nsood on 22/08/16.
 */
public class JavaPipelineClassification {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("JavaPipelineClassification")
                .master("local")
                .getOrCreate();

        // $example on$
        // Prepare training documents, which are labeled.
        Dataset<Row> training = spark.createDataFrame(Arrays.asList(
                new JavaLabeledDocument(0L, "a b c d e spark", 1.0),
                new JavaLabeledDocument(1L, "b d", 0.0),
                new JavaLabeledDocument(2L, "spark f g h", 1.0),
                new JavaLabeledDocument(3L, "hadoop mapreduce", 0.0)
        ), JavaLabeledDocument.class);

        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField(
                        "words", DataTypes.createArrayType(DataTypes.StringType), false, Metadata.empty())
        });

        // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("text")
                .setOutputCol("words");
        Dataset<Row> wordsDataFrame = tokenizer.transform(training);
        for (Row r : wordsDataFrame.select("words", "label").takeAsList(3)) {
            java.util.List<String> words = r.getList(0);
            for (String word : words) System.out.print(word + "|");
            System.out.println();
            System.out.println(r.get(1));
        }
        NGram ngramTransformer = new NGram().setN(2).setInputCol(tokenizer.getOutputCol()).setOutputCol("ngrams");
        Dataset<Row> ngramDataFrame = ngramTransformer.transform(wordsDataFrame);
        HashingTF hashingTF = new HashingTF()
                .setNumFeatures(1000)
                .setInputCol(ngramTransformer.getOutputCol())
                .setOutputCol("features");
        Dataset<Row> output = hashingTF.transform(ngramDataFrame);
        NaiveBayes nb = new NaiveBayes();
        //Model model = nb.fit(output);
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, ngramTransformer, hashingTF, nb});
        PipelineModel model = pipeline.fit(training);

        // Prepare test documents, which are unlabeled.
        Dataset<Row> test = spark.createDataFrame(Arrays.asList(
                new JavaDocument(4L, "spark i j k"),
                new JavaDocument(5L, "l m n"),
                new JavaDocument(6L, "spark hadoop spark"),
                new JavaDocument(7L, "apache hadoop")
        ), JavaDocument.class);

        // Make predictions on test documents.
        Dataset<Row> predictions = model.transform(test);
        for(String col: predictions.columns()) {
            System.out.print(col + " ");
        }
        System.out.println();
        for (Row r : predictions.select("id", "text", "probability", "prediction", "words", "features").collectAsList()) {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> prob=" + r.get(2)
                    + ", prediction=" + r.get(3) + ", words=" + r.get(4) + ", features=" + r.get(5));

        }

        /*VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"features"})
                .setOutputCol("features");
        Dataset<Row> output = assembler.transform(ngramDataFrame);*/

        System.out.println(output.select("features", "label").first());


        /*System.out.println(">>>>>>>ngram columns>>>>");
        for(String col: ngramDataFrame.columns()) {
            System.out.println(">>>>>>>"+ col);
        }
        System.out.println(">>>>>>>ngram columns end>>>>");
        Row[] rows = (Row[])ngramDataFrame.select("label", "features").collect();
        List<Tuple2<Double, List>> trainList = new ArrayList<>();
        List<JavaNaiveInputDocument> docs = new ArrayList<>();
        for(int i = 0; i< rows.length; i++) {
            Row r = rows[i];
            scala.Tuple2 tuple2 = new scala.Tuple2(r.get(0), r.get(1));
            trainList.add(tuple2);
            docs.add(new JavaNaiveInputDocument((double)r.get(0),
                    ((List<String>)((Object)(Arrays.asList(((WrappedArray<Object>)r.get(1)).array()))))));
        }
        System.out.println(">>>>>>>>>>>>>>>tuple>>>>>>");
        for(Tuple2 t: trainList) {
            System.out.println(t);
        }
        System.out.println(">>>>>>>>>>>>>>>tuple>>>>>>");
        System.out.println(">>>>>>>>>>>>>>>docs>>>>>>");
        for(JavaNaiveInputDocument t: docs) {
            System.out.println(t);
        }
        System.out.println(">>>>>>>>>>>>>>>docs end>>>>>>");
        *//*Encoder<JavaNaiveInputDocument> encoder = Encoders.bean(JavaNaiveInputDocument.class);
        Dataset<JavaNaiveInputDocument> ds2 = spark.createDataset(docs, encoder);*//*

        Encoder<Tuple2<Double, List>> encoder = Encoders.tuple(Encoders.DOUBLE(), Encoders.bean(List.class));
        Dataset<Tuple2<Double, List>> ds2 = spark.createDataset(trainList, encoder);

        NaiveBayes nb = new NaiveBayes();
        Model model = nb.fit(ds2);*/
        /*Dataset<Row> test = spark.createDataFrame(Arrays.asList(
                new JavaDocument(4L, "spark i j k"),
                new JavaDocument(5L, "l m n"),
                new JavaDocument(6L, "spark hadoop spark"),
                new JavaDocument(7L, "apache hadoop")
        ), JavaDocument.class);
        model.transform(test);*/
        /*Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, ngramTransformer});
        PipelineModel model = pipeline.fit(training);*/

        // Make predictions on test documents.
        /*Dataset<Row> predictions = model.transform(test);
        for (Row r : predictions.select("id", "text", "words", "features").collectAsList()) {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> words=" + r.get(2) + ", features=" + r.get(3));
        }*/
        /*RDD<Row> trainData = predictions.select("label","features").rdd();
        List<Tuple2<Double, List<String>>> trainList = new ArrayList<>(*//*trainData.count()*//*);
        Row[] rows = (Row[])predictions.select("label","features").collect();
        for(int i = 0; i< rows.length; i++) {
            Row r = rows[i];
            scala.Tuple2 tuple2 = new scala.Tuple2(r.get(0), r.get(1));
            trainList.add(tuple2);
        }
        System.out.println(">>>>>>>>>>>>>>>tuple>>>>>>");
        for(Tuple2 t: trainList) {
            System.out.println(t);
        }
        System.out.println(">>>>>>>>>>>>>>>tuple>>>>>>");
*/
        //)line =>  LabeledPoint(Try(line(0).asInstanceOf[Integer]).get.toDouble,
        // Try(line(1).asInsta nceOf[org.apache.spark.mllib.linalg.SparseVector]).get))

        /*NaiveBayes nb = new NaiveBayes();
        pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, ngramTransformer, nb});
        model = pipeline.fit(training);
        predictions = model.transform(test);
        System.out.println(">>>>>>>>>>>>>");
        for(String col: predictions.columns()) {
            System.out.print(col + " ");
        }
        System.out.println(">>>>>>>>>>>>>");*/
        /*for (Row r : predictions.select("id", "text", "probability", "prediction", "words", "features").collectAsList()) {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> prob=" + r.get(2)
                    + ", prediction=" + r.get(3) + ", words=" + r.get(4) + ", features=" + r.get(5));
        }*/
        /*for (Row r : ngramDataFrame.select("ngrams", "label").takeAsList(3)) {
            java.util.List<String> ngrams = r.getList(0);
            for (String ngram : ngrams) System.out.print(ngram + " --- ");
            System.out.println();
        }*/
/*

        //Dataset<Row> wordDataFrame = spark.createDataFrame(tokenizer.getOutputCol(), schema);

        HashingTF hashingTF = new HashingTF()
                .setNumFeatures(1000)
                .setInputCol(tokenizer.getOutputCol())
                .setOutputCol("features");
        NaiveBayes nb = new NaiveBayes();
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {tokenizer, hashingTF, nb});

        // Fit the pipeline to training documents.
        PipelineModel model = pipeline.fit(training);

        // Prepare test documents, which are unlabeled.
        Dataset<Row> test = spark.createDataFrame(Arrays.asList(
                new JavaDocument(4L, "spark i j k"),
                new JavaDocument(5L, "l m n"),
                new JavaDocument(6L, "spark hadoop spark"),
                new JavaDocument(7L, "apache hadoop")
        ), JavaDocument.class);

        // Make predictions on test documents.
        Dataset<Row> predictions = model.transform(test);
        for (Row r : predictions.select("id", "text", "probability", "prediction").collectAsList()) {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + ") --> prob=" + r.get(2)
                    + ", prediction=" + r.get(3));
        }
        // $example off$
*/

        spark.stop();
    }
}

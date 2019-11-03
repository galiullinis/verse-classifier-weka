import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.util.Random;

public class Main {
    private static void createModel() throws Exception {
        String PATH_TO_ARFF = "PATH_TO_ARFF";
        String PATH_TO_MODEL = "PATH_TO_MODEL";
        ArffLoader arffLoader = new ArffLoader();
        arffLoader.setFile(new File(PATH_TO_ARFF));
        Instances structure = arffLoader.getDataSet();
        structure.setClassIndex(structure.numAttributes() - 1);
        System.out.println(structure.toSummaryString());
        RandomForest rf = new RandomForest();
        rf.setMaxDepth(5);
        rf.setNumFeatures(4);
        rf.buildClassifier(structure);
        Evaluation evaluation = new Evaluation(structure);
        evaluation.crossValidateModel(rf, structure, 10, new Random(1));
        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));
        weka.core.SerializationHelper.write(PATH_TO_MODEL, rf);
    }

    public static void main(String[] args) throws Exception {
//        createModel();
        String test = "0 0 0 0";
        Verse verse = new Verse();
        System.out.println("Classification: " + verse.classifyText(test));
    }
}

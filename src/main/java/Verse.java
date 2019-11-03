import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.*;
import java.util.ArrayList;

public class Verse {
    private Classifier classModel;
    private Instances dataModel;
    private String classModelFile = "src/main/resources/verse.model";

    public Verse() throws Exception {
        InputStream inputStream = new FileInputStream(classModelFile);
        classModel = (Classifier) SerializationHelper.read(inputStream);
    }

    public void close(){
        classModel = null;
        classModelFile = null;
    }

    public String classifyText(String signs){
        ArrayList dataClasses = new ArrayList();
        ArrayList dataAttrs = new ArrayList();
        Attribute types;
        double values[] = getValues(signs);
        int i = 0, maxIndex = 0;

        dataClasses.add("verse");
        dataClasses.add("not_verse");
        types = new Attribute("types", dataClasses);

        dataAttrs.add(new Attribute("text_volume"));
        dataAttrs.add(new Attribute("line_length"));
        dataAttrs.add(new Attribute("punct_to_words"));
        dataAttrs.add(new Attribute("rhyme"));
        dataAttrs.add(types);

        dataModel = new Instances("classify", dataAttrs, 0);
        dataModel.setClass(types);
        dataModel.add(new DenseInstance(1, values));
        dataModel.instance(0).setClassMissing();

        double cl[] = new double[0];
        try {
            cl = classModel.distributionForInstance(dataModel.instance(0));
        } catch (Exception e) {
            e.printStackTrace();
        }
        for(i = 0; i < cl.length; i++)
            if(cl[i] > cl[maxIndex])
                maxIndex = i;

        return dataModel.classAttribute().value(maxIndex);

    }

    private double[] getValues(String signs){
        String[] slist = signs.split("\\s");
        double[] array = new double[slist.length + 1];
        int i = 0;
        for (String sign : slist){
            array[i] = Double.valueOf(sign);
            i++;
        }
        return array;
    }
}

import java.io.Serializable;
import java.util.List;

/**
 * Created by nsood on 23/08/16.
 */
public class JavaNaiveInputDocument implements Serializable {
    private double label;
    private List<String> features;

    public JavaNaiveInputDocument(){}

    public JavaNaiveInputDocument(double label, List<String> features) {
        this.label = label;
        this.features = features;
    }

    public double getLabel() {
        return label;
    }

    public List<String> getFeatures() {
        return features;
    }

    public void setLabel(double label) {
        this.label = label;
    }

    public void setFeatures(List<String> features) {
        this.features = features;
    }
}
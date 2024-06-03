import Activations.Activation;
import Losses.Loss;
import com.opencsv.exceptions.CsvException;
import org.nd4j.shade.guava.base.Stopwatch;

import java.io.IOException;
import java.time.Duration;
import java.util.Arrays;
import java.util.Random;


public class Main {

    public static NeuralNetwork network;
    public NeuralNetwork nn;

    // Define inputs for XOR operation
    public static void main(String[] args) throws IOException, CsvException {
        System.out.println("Hallo world!");
        Main main = new Main();
    }

    public Main() throws IOException {
        XORSolver();
    }


    public void XORSolver() {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] targets = {{1}, {0}, {0}, {1}};
        Stopwatch stopwatch = Stopwatch.createStarted();
        int[] layers = {2, 4, 1};
        network = new NeuralNetwork(2, Activation.Sigmoid,
                4, Activation.Sigmoid,
                1, Activation.Sigmoid);
        network.setLoss(Loss.BinaryCrossEntropy);
        network.toggleAdam();
        int epochs = 100000;
        network.trainNetwork(inputs, targets, epochs);

        System.out.println("Predictions after training:");
        for (double[] input : inputs) {
            double output = network.predict(input)[0];

            double roundedOutput = Math.round(output);
            System.out.println(Arrays.toString(input) + " -> " + roundedOutput + "\t(" + output + ")");
        }

        System.out.println("Successfully Trained the Network!");
        stopwatch.stop();
        Duration duration = stopwatch.elapsed();

        System.out.println("It took: " + (duration.toMillis() / 1000L) + "seconds, and " + (duration.toNanos() % 1000000L) + " Nano-Seconds!");
//        Model.createNewFile("myModel");
//        network.save("C:\\Users\\Or\\IdeaProjects\\MachineLearning\\src\\Saves\\SavedModel.json");
//        JsonHandler.createFile("C:\\Users\\Or\\IdeaProjects\\MachineLearning\\src\\Saves\\SavedModel.obj");
//        JsonHandler.writeToFile("C:\\Users\\Or\\IdeaProjects\\MachineLearning\\src\\Saves\\SavedModel.obj", network);
    }


    public static double[][] scaleFeatures(double[][] features) {
        int numRows = features.length;
        int numCols = features[0].length;

        // Calculate mean and standard deviation for each feature
        double[] mean = new double[numCols];
        double[] stdDev = new double[numCols];

        for (int j = 0; j < numCols; j++) {
            double sum = 0.0;
            for (int i = 0; i < numRows; i++) {
                sum += features[i][j];
            }
            mean[j] = sum / numRows;

            double sumSquaredDiff = 0.0;
            for (int i = 0; i < numRows; i++) {
                sumSquaredDiff += Math.pow(features[i][j] - mean[j], 2);
            }
            stdDev[j] = Math.sqrt(sumSquaredDiff / numRows);
        }

        // Scale features using z-score normalization
        double[][] scaledFeatures = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                scaledFeatures[i][j] = (features[i][j] - mean[j]) / stdDev[j];
            }
        }

        return scaledFeatures;
    }

    public static double[][] standardizeData(double[][] data) {
        int numSamples = data.length;
        int numFeatures = data[0].length;

        // Compute mean and standard deviation for each feature
        double[] mean = new double[numFeatures];
        double[] stdDeviation = new double[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            double sum = 0;
            for (int i = 0; i < numSamples; i++) {
                sum += data[i][j];
            }
            mean[j] = sum / numSamples;

            double sumSquaredDiff = 0;
            for (int i = 0; i < numSamples; i++) {
                double diff = data[i][j] - mean[j];
                sumSquaredDiff += diff * diff;
            }
            double variance = sumSquaredDiff / numSamples;
            stdDeviation[j] = Math.sqrt(variance);
        }

        // Standardize the data
        double[][] standardizedData = new double[numSamples][numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                standardizedData[i][j] = (data[i][j] - mean[j]) / stdDeviation[j];
            }
        }

        return standardizedData;
    }

    public static void shuffle(double[][] data, double[][] labels) {
        if (data.length != labels.length) {
            throw new IllegalArgumentException("Data and labels must have the same number of samples.");
        }

        int numSamples = data.length;
        Random random = new Random();

        // Combine data and labels into a single array
        double[][] combinedData = new double[numSamples][data[0].length + labels[0].length];
        for (int i = 0; i < numSamples; i++) {
            System.arraycopy(data[i], 0, combinedData[i], 0, data[i].length);
            System.arraycopy(labels[i], 0, combinedData[i], data[i].length, labels[0].length);
        }

        // Shuffle the combined data
        for (int i = 0; i < numSamples - 1; i++) {
            int swapIndex = random.nextInt(numSamples - i) + i;
            double[] tempData = combinedData[i];
            combinedData[i] = combinedData[swapIndex];
            combinedData[swapIndex] = tempData;
        }

        // Separate shuffled data and labels
        for (int i = 0; i < numSamples; i++) {
            System.arraycopy(combinedData[i], 0, data[i], 0, data[i].length);
            System.arraycopy(combinedData[i], data[i].length, labels[i], 0, labels[0].length);
        }
    }
}





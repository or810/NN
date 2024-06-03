import Activations.Activation;
import Losses.*;


import java.io.Serializable;
import java.util.Arrays;


public class NeuralNetwork implements Serializable {

    private int[] layersConfig;
    private transient Activation[] activationsConfig;
    private Dense[] layers;
    private int t = 0;
    private boolean useAdam;
    private double learningRate = 0.3;

    private Loss loss = Loss.BinaryCrossEntropy;

    public NeuralNetwork(Object... sizeAndActivation) {
        if (sizeAndActivation.length % 2 != 0)
            throw new RuntimeException("Invalid input for NeuralNetwork Constructor");

        layersConfig = new int[sizeAndActivation.length / 2];
        activationsConfig = new Activation[sizeAndActivation.length / 2];
        int count = 0;

        for (int i = 0; i < sizeAndActivation.length; i++) {
            if (sizeAndActivation[i] instanceof Integer) {
                layersConfig[count] = (int) sizeAndActivation[i];
                activationsConfig[count++] = (Activation) sizeAndActivation[i + 1];
            } else if (!(sizeAndActivation[i] instanceof Activation))
                throw new RuntimeException("Invalid input for NeuralNetwork Constuctor");
        }

        layers = new Dense[layersConfig.length];
        initializeLayers();
    }



    private void initializeLayers() {

        layers[layers.length - 1] = new Dense(layersConfig[layers.length - 1], activationsConfig[layers.length - 1]);
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Dense(layersConfig[i], activationsConfig[i]);
        }

        combineLayers();
    }

    private double[] feedForward(double[] input) {
        layers[0].setActivations(input);
        // Loop through each layer to calculate activations
        for (int i = 1; i < layers.length; i++)
            input = layers[i].feedForward(input);

        return layers[layers.length - 1].getActivations(); // Return activations of all layers
    }

    public double[] predict(double[] input) {
        return feedForward(input); // Return output layer activations
    }



    public void trainNetwork(double[][] inputs, double[][] targets, int epochs) {
        int presentage = 0;

        for (int epoch = 0, count = 0; epoch < epochs; epoch++, count++) {
            double totalError = 0;
            for (int i = 0; i < inputs.length; i++) {
                train(inputs[i], targets[i]);
                double[] output = feedForward(inputs[i]);
                // Calculate total error for the epoch
                totalError += Arrays.stream(targets[i])
                        .map(j -> Math.abs(j - output[0]))
                        .sum();
            }
            double progress = ((double) (epoch + 1) / epochs) * 100; // Fixing progress calculation
            if (progress >= presentage + 1.0) { // Checking if progress increased by 1%
                presentage = (int) progress;
                System.out.println(totalError);
                //int roundedEpoch = (int) MathUtils.roundDouble((double) (epoch + 1), -1 * (int)(Math.log10(epoch) - 1));
                String message = String.format("Epoch: %-15s Loss: %-15.2e Progress: %-3d%%", epoch, totalError, presentage);
                System.out.println(message);
            }
        }
    }


    private void train(double[] input, double[] target) {
        double[] outputOutput = feedForward(input); // Output of the network
        double[] errors = loss.calcLoss(input, target, outputOutput);
        if (useAdam)
            layers[layers.length - 2].adamBackpropagation(errors, this);
        else
            layers[layers.length - 2].backPropagate(errors, this);
    }


    private void splitLayers() {
        for (int i = 0; i < layers.length; i++) {
            layers[i].split();
        }
    }

    private void combineLayers() {
        layers[0].setNeighbors(null, layers[1]);
        layers[layers.length - 1].setNeighbors(layers[layers.length - 2], null);
        for (int i = 1; i < layers.length - 1; i++) {
            layers[i].setNeighbors(layers[i - 1], layers[i + 1]);
        }
    }

    private double relu(double x) {
        return Math.max(0, x);
    }

    private double reluDerivative(double x) {
        return ((x > 0) ? 1 : 0);
    }

    // Sigmoid activation function
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of sigmoid activation function
    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public void toggleAdam() {
        useAdam = !useAdam;
    }

    public int getT() {
        return t;
    }

    public void addT() {
        t++;
    }

    public void setLoss(Loss loss) {
        this.loss = loss;
    }

    public boolean isUseAdam() {
        return useAdam;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}

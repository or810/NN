import Activations.*;
import com.google.gson.annotations.Expose;
import org.apache.commons.math3.linear.MatrixUtils;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

public class Dense implements Serializable {
    private static final double beta1 = 0.9; // Decay rate for the first moment estimate
    private static final double beta2 = 0.99; // Decay rate for the second moment estimate
    private static final double epsilon = 1e-8; // Smoothing term to avoid division by zero


    private double[][] weights;
    private double[][] m;
    private double[][] v;

    public double bias; // should consider making it an Array.
    private double[] activations;

    private int length;
    @Expose(serialize = false, deserialize = false)
    private Dense prev;
    @Expose(serialize = false, deserialize = false)
    private Dense next;


    private Activation activationFunc;


    // does not initialize everything yet
    public Dense(int size, Activation activationFunc) {
        length = size;
        activations = new double[size];
        this.activationFunc = activationFunc;
    }


    private void randomize() {
        Random random = new Random();
        bias = random.nextDouble();
        for(int i = 0;i<length;i++) {
            for(int j = 0;j<next.length;j++) {
                weights[i][j] = random.nextDouble() * 5;
            }
        }
    }

    // done for every layer except the first one.
    public double[] feedForward(double[] input) {
        double[] output = new double[length];
        for(int i = 0;i<length;i++) { // for every neuron:
            double sum = prev.bias;
            for(int j = 0;j<prev.length;j++) { // for every neuron in prev layer:
                sum += prev.activations[j] * prev.weights[j][i];
            }
            if(!activationFunc.equals(Activation.Softmax))
                activations[i] = activationFunc.apply(sum);
        }
        if(activationFunc.equals(Activation.Softmax)) {
            activations = activationFunc.applySoftmax(activations);
        }
        return activations;
    }

    public void setNeighbors(Dense prev, Dense next) {
        this.prev = prev;
        if(next != null) {
            this.next = next;
            weights = new double[length][next.length];
            m = new double[length][next.length];
            v = new double[length][next.length];
            randomize();
        }
    }

    public void backPropagate(double[] prevError, NeuralNetwork network) {
        double[] errors = new double[length];
        for(int i = 0;i<length;i++) { // for each neuron in this layer.
            double errorSum = 0;
            for(int j = 0;j< next.length;j++) { // for each neuron in next layer.
                errorSum += prevError[j] * weights[i][j];
            }
            errors[i] = errorSum * activationFunc.applyDerivative(activations[i]);
        }

        for(int i = 0;i<length;i++) {
            for(int j = 0;j< next.length;j++) {
                weights[i][j] += network.getLearningRate() * prevError[j] * activations[i];
            }
        }
        bias += network.getLearningRate() * Arrays.stream(prevError).sum();
        if(prev == null)
            return;
        prev.backPropagate(errors, network);
    }


    public void adamBackpropagation(double[] prevError, NeuralNetwork network) {
        double[] errors = new double[length];
        for (int i = 0; i < length; i++) { // for each neuron in this layer.
            double errorSum = 0;
            for (int j = 0; j < next.length; j++) { // for each neuron in next layer.
                errorSum += prevError[j] * weights[i][j];
            }
            errors[i] = errorSum * activationFunc.applyDerivative(activations[i]);
        }

        network.addT();
        for (int i = 0; i < length; i++) { // for each neuron in this layer.
            for (int j = 0; j < next.length; j++) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * prevError[j] * activations[i];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * Math.pow(prevError[j] * activations[i], 2);

                double mHat = m[i][j] / (1 - Math.pow(beta1, network.getT()));
                double vHat = v[i][j] / (1 - Math.pow(beta2, network.getT()));

                weights[i][j] += network.getLearningRate() * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
        bias += network.getLearningRate() * Arrays.stream(prevError).sum();
        if(prev == null)
            return;
        prev.adamBackpropagation(errors, network);
    }


    public double[] getActivations() {
        return activations;
    }

    public void setActivations(double[] activations) {
        this.activations = activations;
    }

    public void setActivationFunc(Activation activationFunc) {
        this.activationFunc = activationFunc;
    }

    public void split() {
        next = null;
        prev = null;
    }



}

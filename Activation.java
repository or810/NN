package Activations;

import java.util.Arrays;

public enum Activation {
    Linear,
    Relu,
    LeakyRelu,
    Sigmoid,
    Softmax;
    // soft max:

    public double apply(double x) {
        switch (this) {
            case Linear -> {
                return x;
            }
            case Relu -> {
                return Math.max(0,x);
            }
            case LeakyRelu -> {
                return Math.max(0.01 * x, x);
            }
            case Sigmoid -> {
                return 1 / (1 + Math.exp(-x));
            }
            default -> {return x;}
        }
    }

    public double applyDerivative(double x) {
        switch (this) {
            case Linear -> {
                return 1;
            }
            case Relu -> {
                return Math.max(0,1);
            }
            case LeakyRelu -> {
                return Math.max(0.01, 1);
            }
            case Sigmoid -> {
                return x * (1 - x);
            }
            default -> {return 1;}
        }
    }

    public double applySoftMaxDerivative(double x) {
        double temperature = 1;
        return x * (1 - x) / temperature;
    }

    public double[] applySoftmax(double[] input) {
        double temperature = 1;
        double sum = Arrays.stream(input).map(i -> Math.exp(i / temperature)).sum();

        return Arrays.stream(input).map(i -> Math.exp(i) / sum).toArray();
    }
}

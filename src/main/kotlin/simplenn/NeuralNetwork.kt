package learning.toni.simplenn

import java.io.File
import java.util.Random;


class NeuralNetwork(
    private val inputLayerSize: Int,
    private val hiddenLayerSize: Int,
    private val outputLayerSize: Int,
    private val learningRate: Double
) {
    private val weightsInputToHidden = Array(inputLayerSize) { DoubleArray(hiddenLayerSize) { 0.0 } }
    private val weightsHiddenToOutput = Array(hiddenLayerSize) { DoubleArray(outputLayerSize) { 0.0 } }

    private var inputs = DoubleArray(inputLayerSize)
    private var hiddenLayer = DoubleArray(hiddenLayerSize)
    private var output = 0.0

    init {
        initializeWeights()
    }



    /**
     * Initializes the weights of the Neural Network.
     *
     * The method calculates the standard deviations for the input-to-hidden and hidden-to-output weights.
     * It then uses a random number generator to assign random weights to the weights matrices.
     *
     * @param None
     * @return None
     */
    private fun initializeWeights() {
        // Calc standard deviation for input-to-hidden and hidden-to-output weights
        // This is important because it helps to prevent the weights from being too large or too small
        val inputHiddenStd = Math.sqrt(2.0 / (inputLayerSize + hiddenLayerSize))
        val hiddenOutputStd = Math.sqrt(2.0 / (hiddenLayerSize + outputLayerSize))

        val random = Random()

        // Assign random weights to the weights matrices for the Input to Hidden Layers
        weightsInputToHidden.forEachIndexed { i, weights ->
            weights.forEachIndexed { j, _ ->
                weights[j] = random.nextGaussian() * inputHiddenStd
            }
        }

        // Assign random weights to the weights matrices for the Hidden to Output Layers
        weightsHiddenToOutput.forEachIndexed { i, weights ->
            weights.forEachIndexed { j, _ ->
                weights[j] = random.nextGaussian() * hiddenOutputStd
            }
        }
    }


    fun forward(input: DoubleArray): Double {
        // Update the inputs
        inputs = input

        // Calculate the hidden layer
        hiddenLayer = DoubleArray(hiddenLayerSize) { i ->
            inputs
                .mapIndexed { j, input -> input * weightsInputToHidden[j][i] }
                .sum()
                .sigmoid()
        }

        // Use the hidden layer to calculate the output, activating it with the sigmoid function
        output = hiddenLayer
            .mapIndexed { i, hidden -> hidden * weightsHiddenToOutput[i][0] }
            .sum()
            .sigmoid()

        return output
    }

    fun backpropagate(expected: Double) {
        val outputError = expected - output
        val outputDelta = outputError * output.sigmoidDerivative()

        val hiddenErrors = weightsHiddenToOutput.map { it[0] * outputDelta }
        val hiddenDeltas = hiddenErrors.mapIndexed { i, error -> error * hiddenLayer[i].sigmoidDerivative() }

        weightsHiddenToOutput.forEachIndexed { i, weights ->
            weights.forEachIndexed { j, _ ->
                weights[j] += learningRate * outputDelta * hiddenLayer[i]
            }
        }

        weightsInputToHidden.forEachIndexed { i, weights ->
            weights.forEachIndexed { j, _ ->
                weights[j] += learningRate * hiddenDeltas[j] * inputs[i]
            }
        }
    }

    fun train(data: List<Pair<DoubleArray, Double>>, epochs: Int) {
        repeat(epochs) { epoch ->
            println("Epoch $epoch")
            data.forEach { (input, expected) ->
                forward(input)
                backpropagate(expected)
            }
        }
    }

    fun predict(input: DoubleArray): Double {
        return forward(input)
    }


    fun getWeights(): Pair<Array<DoubleArray>, Array<DoubleArray>> {
        return Pair(weightsInputToHidden, weightsHiddenToOutput)
    }


    fun saveJsonModel(path: String) {
        val weightsInputToHiddenString = weightsInputToHidden.joinToString("\n,") { it.joinToString(",") }
        val weightsHiddenToOutputString = weightsHiddenToOutput.joinToString("\n,") { it.joinToString(",") }

        File(path).writeText(
            """
            {
                "weightsInputToHidden": [
                    $weightsInputToHiddenString
                ],
                "weightsHiddenToOutput": [
                    $weightsHiddenToOutputString
                ]
            }
            """.trimIndent()
        )
    }



}

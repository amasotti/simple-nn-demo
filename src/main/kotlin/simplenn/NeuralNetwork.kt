package learning.toni.simplenn

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import simplenn.Weights
import java.io.File
import java.util.Random

// Neural Network class
class NeuralNetwork(
    private val inputLayerSize: Int,
    private val hiddenLayerSize: Int,
    private val outputLayerSize: Int,
    private val learningRate: Double
) {
    private val weightsInputToHidden = Array(inputLayerSize) { DoubleArray(hiddenLayerSize) }
    private val weightsHiddenToOutput = Array(hiddenLayerSize) { DoubleArray(outputLayerSize) }

    private var inputs = DoubleArray(inputLayerSize)
    private var hiddenLayer = DoubleArray(hiddenLayerSize)
    private var output = 0.0

    init {
        initializeWeights()
    }

    private fun initializeWeights() {
        val inputHiddenStd = Math.sqrt(2.0 / (inputLayerSize + hiddenLayerSize))
        val hiddenOutputStd = Math.sqrt(2.0 / (hiddenLayerSize + outputLayerSize))
        val random = Random()

        weightsInputToHidden.forEach { weights ->
            weights.indices.forEach { j ->
                weights[j] = random.nextGaussian() * inputHiddenStd
            }
        }

        weightsHiddenToOutput.forEach { weights ->
            weights.indices.forEach { j ->
                weights[j] = random.nextGaussian() * hiddenOutputStd
            }
        }
    }

    fun forward(input: DoubleArray): Double {
        inputs = input

        hiddenLayer = DoubleArray(hiddenLayerSize) { i ->
            inputs
                .mapIndexed { j, input -> input * weightsInputToHidden[j][i] }
                .sum()
                .sigmoid()
        }

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
            weights.indices.forEach { j ->
                weights[j] += learningRate * outputDelta * hiddenLayer[i]
            }
        }

        weightsInputToHidden.forEachIndexed { i, weights ->
            weights.indices.forEach { j ->
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

    fun predict(input: DoubleArray): Double = forward(input)

    fun saveJsonModel(path: String) {
        val weights = Weights(weightsInputToHidden, weightsHiddenToOutput)
        val json = Json.encodeToString(weights)
        File(path).writeText(json)
    }

    fun loadWeights(weights: Weights) {
        weights.inputToHidden.forEachIndexed { i, row ->
            row.forEachIndexed { j, value ->
                weightsInputToHidden[i][j] = value
            }
        }
        weights.hiddenToOutput.forEachIndexed { i, row ->
            row.forEachIndexed { j, value ->
                weightsHiddenToOutput[i][j] = value
            }
        }
    }

    companion object {
        fun loadJsonModel(path: String): NeuralNetwork {
            val json = File(path).readText()
            val weights = Json.decodeFromString<Weights>(json)
            return NeuralNetwork(
                weights.inputLayerSize(),
                weights.hiddenLayerSize(),
                weights.outputLayerSize(),
                0.1
            ).apply {
                loadWeights(weights)
            }
        }
    }
}

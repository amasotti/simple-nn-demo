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


    private fun initializeWeights() {
        val inputHiddenStd = Math.sqrt(2.0 / (inputLayerSize + hiddenLayerSize))
        val hiddenOutputStd = Math.sqrt(2.0 / (hiddenLayerSize + outputLayerSize))

        val randomNumber = Random()

        for (i in 0 until inputLayerSize) {
            for (j in 0 until hiddenLayerSize) {
                weightsInputToHidden[i][j] = randomNumber.nextGaussian() * inputHiddenStd
            }
        }


        for (i in 0 until hiddenLayerSize) {
            weightsHiddenToOutput[i] = DoubleArray(outputLayerSize) { randomNumber.nextGaussian() * hiddenOutputStd }
        }
    }


    fun forward(input: DoubleArray): Double {
        inputs = input
        hiddenLayer = DoubleArray(hiddenLayerSize) { i -> sigmoid(
            inputs.zip(weightsInputToHidden.map { it[i] }).sumOf { it.first * it.second }) }
        output = sigmoid(hiddenLayer.zip(weightsHiddenToOutput).sumOf { it.first * it.second[0] })
        return output
    }

    fun backpropagate(expected: Double) {
        val outputError = expected - output
        val outputDelta = outputError * sigmoidDerivative(output)

        val hiddenError = weightsHiddenToOutput.map { it[0] * outputDelta }.toDoubleArray()
        val hiddenDelta = hiddenError.zip(hiddenLayer).map { it.first * sigmoidDerivative(it.second) }.toDoubleArray()

        for (i in 0 until hiddenLayerSize) {
            for (j in 0 until outputLayerSize) {
                weightsHiddenToOutput[i][j] += learningRate * outputDelta * hiddenLayer[i]
            }
        }

        for (i in 0 until inputLayerSize) {
            for (j in 0 until hiddenLayerSize) {
                weightsInputToHidden[i][j] += learningRate * hiddenDelta[j] * inputs[i]
            }
        }
    }

    fun train(data: List<Pair<DoubleArray, Double>>, epochs: Int) {
        for (epoch in 0 until epochs) {
            println("Epoch $epoch")
            for ((input, expected) in data) {
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
        val weights = getWeights()
        val weightsInputToHidden = weights.first
        val weightsHiddenToOutput = weights.second

        val weightsInputToHiddenString = weightsInputToHidden.joinToString("\n") { it.joinToString(",") }
        val weightsHiddenToOutputString = weightsHiddenToOutput.joinToString("\n") { it.joinToString(",") }

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

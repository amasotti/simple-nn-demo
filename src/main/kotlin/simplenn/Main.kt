package learning.toni.simplenn

fun main() {
    val neuralNetwork = NeuralNetwork(inputLayerSize = 2, hiddenLayerSize = 2, outputLayerSize = 1, learningRate = 0.1)

    // Training data for OR problem
    val trainingData = listOf(
        doubleArrayOf(0.0, 0.0) to 0.0,
        doubleArrayOf(0.0, 1.0) to 1.0,
        doubleArrayOf(1.0, 0.0) to 1.0,
        doubleArrayOf(1.0, 1.0) to 1.0
    )

    neuralNetwork.train(trainingData, epochs = 10000)

    neuralNetwork.saveJsonModel("model.json")

    val trainedModel = NeuralNetwork.loadJsonModel("model.json")


    // Testing the network
    val testData = listOf(
        doubleArrayOf(0.0, 0.0),
        doubleArrayOf(0.0, 1.0),
        doubleArrayOf(1.0, 0.0),
        doubleArrayOf(1.0, 1.0)
    )

    testData.forEach { input ->
        val prediction = trainedModel.predict(input)
        println("Input: ${input.joinToString(", ")} -> Prediction: $prediction")
    }


}

package simplenn

import kotlinx.serialization.Serializable

@Serializable
data class Weights(
    val inputToHidden: Array<DoubleArray>,
    val hiddenToOutput: Array<DoubleArray>
) {
    override fun toString(): String {
        return "Weights(inputToHidden=${inputToHidden.contentDeepToString()}, hiddenToOutput=${hiddenToOutput.contentDeepToString()})"
    }

    fun inputLayerSize() = inputToHidden.size
    fun hiddenLayerSize() = hiddenToOutput.size
    fun outputLayerSize() = hiddenToOutput[0].size


}

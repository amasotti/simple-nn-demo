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
}

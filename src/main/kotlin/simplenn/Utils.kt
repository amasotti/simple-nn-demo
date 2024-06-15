package learning.toni.simplenn

fun sigmoid (x: Double): Double {
    return 1 / (1 + Math.exp(-x))
}

fun sigmoidDerivative (x: Double): Double {
    return x * (1 - x)
}

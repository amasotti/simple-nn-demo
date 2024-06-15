package learning.toni.simplenn

import kotlin.math.exp

// Extension function for the sigmoid activation function
fun Double.sigmoid() = 1.0 / (1.0 + exp(-this))

// Extension function for the derivative of the sigmoid function
fun Double.sigmoidDerivative() = this * (1.0 - this)

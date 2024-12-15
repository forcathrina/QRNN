from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.primitives import BackendEstimatorV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import nn, stack, arctan
from math import sqrt



class QNN(nn.Module):
    def __init__(self, sequence_length, gates=(RYGate, RXGate, RZGate)):
        super().__init__()

        self.gates = gates
        self.sequence_length = sequence_length

        # Parameter vectors for input and trained weights
        self.input_params = ParameterVector('Input', self.sequence_length)

        self.trainable_params = ParameterVector('Weights', self.sequence_length * 12)

        # Define observables for measurement
        self.observables = [
            SparsePauliOp('Z')
        ]

        # Quantum Neural Network setup
        backend_estimator = BackendEstimatorV2(backend=GenericBackendV2(num_qubits=2))
        self.qnn = EstimatorQNN(
            circuit=self.build_quantum_circuit(),
            observables=self.observables,
            input_params=self.input_params.params,
            weight_params=self.trainable_params.params,
            estimator=backend_estimator,
            gradient=ParamShiftEstimatorGradient(backend_estimator, pass_manager=PassManager())
        )

        self.qnn_model = TorchConnector(self.qnn)

    def build_quantum_circuit(self):
        qc = QuantumCircuit(1)
        for seq in range(self.sequence_length):
            qc.append(RYGate(self.input_params[seq]), [0])
        
            qc.append(self.gates[0](self.trainable_params[seq * 12]), [0])
            qc.append(self.gates[1](self.trainable_params[seq * 12 + 1]), [0])
            qc.append(self.gates[2](self.trainable_params[seq * 12 + 2]), [0])
            
            # 제곱 근사 회로
            qc.append(RYGate(self.input_params[seq]**2), [0])
            
            qc.append(self.gates[0](self.trainable_params[seq * 12 + 3]), [0])
            qc.append(self.gates[1](self.trainable_params[seq * 12 + 4]), [0])
            qc.append(self.gates[2](self.trainable_params[seq * 12 + 5]), [0])
            
            # 세제곱 근사 회로
            qc.append(RYGate(self.input_params[seq]**3), [0])

            qc.append(self.gates[0](self.trainable_params[seq * 12 + 6]), [0])
            qc.append(self.gates[1](self.trainable_params[seq * 12 + 7]), [0])
            qc.append(self.gates[2](self.trainable_params[seq * 12 + 8]), [0])
            
            # 네제곱 근사 회로
            qc.append(RYGate(self.input_params[seq]**4), [0])
            
            qc.append(self.gates[0](self.trainable_params[seq * 12 + 9]), [0])
            qc.append(self.gates[1](self.trainable_params[seq * 12 + 10]), [0])
            qc.append(self.gates[2](self.trainable_params[seq * 12 + 11]), [0])
            
            qc.barrier()
        return qc

    def forward(self, x):
        """
        Processes input of shape (batch_size, sequence_length, input_size).
        Each batch is processed as a sequence of length (sequence_length,) using qnn_model.
        Final output has shape (batch_size, 1) where each batch produces a weighted sum of measurements.
        """
        # Input x has shape (batch_size, sequence_length, input_size)
        x = arctan(x)  # Apply arctan element-wise transformation
        # Shape of x remains (batch_size, sequence_length, input_size)

        batch_size, sequence_length, input_size = x.size()
        outputs = []


        for b in range(batch_size):  # Process each batch separately
            single_batch = x[b]  # Shape: (sequence_length, input_size)

            # Reshape single_batch for qnn_model
            # Flatten along the sequence dimension to match (sequence_length,)
            flattened_sequence = single_batch.view(sequence_length, input_size)  # Shape: (sequence_length,)

            # Forward pass through qnn_model
            batch_output = self.qnn_model.forward(flattened_sequence)  # Shape: (input_size,)

            outputs.append(batch_output)

        # Stack outputs to create (batch_size, input_size)
        
        outputs = stack(outputs, dim=0)  # Shape: (batch_size, input_size)

        return outputs[:, -1 , :]

        
class QNN_Tayler(nn.Module):
    def __init__(self, sequence_length, gates=(RYGate, RXGate, RZGate)):
        super().__init__()

        self.gates = gates
        self.sequence_length = sequence_length

        # Parameter vectors for input and trained weights
        self.input_params = ParameterVector('Input', self.sequence_length)

        self.trainable_params = ParameterVector('Weights', self.sequence_length * 12)

        # Define observables for measurement
        self.observables = [
            SparsePauliOp('Z')
        ]

        # Quantum Neural Network setup
        backend_estimator = BackendEstimatorV2(backend=GenericBackendV2(num_qubits=2))
        self.qnn = EstimatorQNN(
            circuit=self.build_quantum_circuit(),
            observables=self.observables,
            input_params=self.input_params.params,
            weight_params=self.trainable_params.params,
            estimator=backend_estimator,
            gradient=ParamShiftEstimatorGradient(backend_estimator, pass_manager=PassManager())
        )

        self.qnn_model = TorchConnector(self.qnn)

    def build_quantum_circuit(self):
        qc = QuantumCircuit(1)
        for seq in range(self.sequence_length):
            qc.append(RYGate(self.input_params[seq]), [0])
        
            qc.append(self.gates[0](self.trainable_params[seq * 12]), [0])
            qc.append(self.gates[1](self.trainable_params[seq * 12 + 1]), [0])
            qc.append(self.gates[2](self.trainable_params[seq * 12 + 2]), [0])
            

            # Approximate X^2 using RY and RX rotations
            qc.append(RYGate(self.input_params[seq] * sqrt(2)), [0])
            qc.append(RXGate(self.input_params[seq] * sqrt(2)), [0])
            qc.append(RZGate(self.input_params[seq] * sqrt(2)), [0])
            # Based on Taylor expansion, cos^2(x^2/2) can be approximated as 
            # cos^2(alpha/2)cos^2(beta/2) - sin^2(alpha/2)sin^2(beta/2)
            
            qc.append(self.gates[0](self.trainable_params[seq * 12 + 3]), [0])
            qc.append(self.gates[1](self.trainable_params[seq * 12 + 4]), [0])
            qc.append(self.gates[2](self.trainable_params[seq * 12 + 5]), [0])
            
            # 세제곱 근사 회로
            qc.append(RYGate(self.input_params[seq] * sqrt(3)), [0])
            qc.append(RXGate(self.input_params[seq] * sqrt(3)), [0])
            qc.append(RZGate(self.input_params[seq] * sqrt(3)), [0])
            
            qc.append(self.gates[0](self.trainable_params[seq * 12 + 6]), [0])
            qc.append(self.gates[1](self.trainable_params[seq * 12 + 7]), [0])
            qc.append(self.gates[2](self.trainable_params[seq * 12 + 8]), [0])
            
            # 네제곱 근사 회로
            qc.append(RYGate(self.input_params[seq] * sqrt(4)), [0])
            qc.append(RXGate(self.input_params[seq] * sqrt(4)), [0])
            qc.append(RZGate(self.input_params[seq] * sqrt(4)), [0])
            
            qc.append(self.gates[0](self.trainable_params[seq * 12 + 9]), [0])
            qc.append(self.gates[1](self.trainable_params[seq * 12 + 10]), [0])
            qc.append(self.gates[2](self.trainable_params[seq * 12 + 11]), [0])
            
            qc.barrier()
        return qc

    def forward(self, x):
        """
        Processes input of shape (batch_size, sequence_length, input_size).
        Each batch is processed as a sequence of length (sequence_length,) using qnn_model.
        Final output has shape (batch_size, 1) where each batch produces a weighted sum of measurements.
        """
        # Input x has shape (batch_size, sequence_length, input_size)
        x = arctan(x)  # Apply arctan element-wise transformation
        # Shape of x remains (batch_size, sequence_length, input_size)

        batch_size, sequence_length, input_size = x.size()
        outputs = []


        for b in range(batch_size):  # Process each batch separately
            single_batch = x[b]  # Shape: (sequence_length, input_size)

            # Reshape single_batch for qnn_model
            # Flatten along the sequence dimension to match (sequence_length,)
            flattened_sequence = single_batch.view(sequence_length, input_size)  # Shape: (sequence_length,)

            # Forward pass through qnn_model
            batch_output = self.qnn_model.forward(flattened_sequence)  # Shape: (input_size,)

            outputs.append(batch_output)

        # Stack outputs to create (batch_size, input_size)
        
        outputs = stack(outputs, dim=0)  # Shape: (batch_size, input_size)

        return outputs[:, -1 , :]

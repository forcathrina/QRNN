import torch
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


class QRNN(torch.nn.Module):
    def __init__(self, sequence_length, num_qubits=1, gates=(RYGate, RXGate, RZGate), initial_params=(0, 0)):
        super().__init__()

        self.gates = gates
        self.sequence_length = sequence_length
        self.num_qubits = num_qubits

        # Initial random parameters for training
        self.initial_params = torch.nn.Parameter(torch.tensor(initial_params, dtype=torch.float32))

        # Parameter vectors for input and trained weights
        self.input_params = ParameterVector('Input', self.sequence_length)

        self.trainable_params = ParameterVector('Weights', num_qubits * 2)

        # Define observable for measurement
        self.observable = SparsePauliOp('Z')

        backend_estimator = BackendEstimatorV2(backend=GenericBackendV2(num_qubits=2))

        # Quantum Neural Network setup
        self.qrnn = EstimatorQNN(
            circuit=self.build_quantum_circuit(),
            observables=[self.observable],
            input_params=self.input_params.params,
            weight_params=self.trainable_params.params,
            num_virtual_qubits=num_qubits,
            estimator=backend_estimator,
            gradient=ParamShiftEstimatorGradient(backend_estimator, pass_manager=PassManager())
        )

        self.qrnn_model = TorchConnector(self.qrnn, initial_weights=self.initial_params)

    def build_quantum_circuit(self):
        qc = QuantumCircuit(self.num_qubits)

        for qubit_index in range(self.num_qubits):
            for seq in range(self.sequence_length):
                qc.append(self.gates[0](self.input_params[seq]), [qubit_index])
                qc.append(self.gates[1](self.trainable_params[qubit_index * 2]), [qubit_index])
                qc.append(self.gates[2](self.trainable_params[qubit_index * 2 + 1]), [qubit_index])
                qc.barrier()

        return qc

    def forward(self, x):
        """
        Processes input of shape (batch_size, sequence_length, input_size).
        Each batch is processed as a sequence of length (sequence_length,) using qrnn_model.
        Final output has shape (batch_size, input_size).
        """
        x = torch.arctan(x)  # Apply arctan element-wise transformation

        batch_size, sequence_length, input_size = x.size()
        outputs = []

        for b in range(batch_size):  # Process each batch separately
            single_batch = x[b]  # Shape: (sequence_length, input_size)

            # Reshape single_batch for qrnn_model
            # Flatten along the sequence dimension to match (sequence_length,)
            flattened_sequence = single_batch.view(sequence_length)  # Shape: (sequence_length,)

            # Forward pass through qrnn_model
            batch_output = self.qrnn_model.forward(flattened_sequence)  # Shape: (input_size,)

            outputs.append(batch_output)

        # Stack outputs to create (batch_size, input_size)
        outputs = torch.stack(outputs, dim=0)  # Shape: (batch_size, input_size)

        return outputs

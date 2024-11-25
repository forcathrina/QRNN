import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class ClassicalMappedQRNN(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 4, alpha: float = np.pi/2, beta: float = np.pi/2):
        """
        Args:
            input_size: 입력 차원 (기본값 1, 각 시점의 입력이 단일 각도값인 경우)
            hidden_size: 4로 고정 (quantum state의 실수부/허수부를 표현하기 위해)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # 학습 가능한 alpha, beta 파라미터
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32, requires_grad=True))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32, requires_grad=True))

    def _compute_rx_matrix(self) -> torch.Tensor:
        """
        Rx(alpha) 계산:
        [cos(α/2),    -i*sin(α/2)]
        [-i*sin(α/2), cos(α/2)   ]

        → 4x4 실수 행렬로 변환:
        [cos(α/2),  0,        0,         -sin(α/2)]
        [0,         cos(α/2), -sin(α/2),  0       ]
        [0,         sin(α/2), cos(α/2),   0       ]
        [sin(α/2),  0,        0,         cos(α/2) ]
        """
        alpha = self.alpha
        c = torch.cos(alpha / 2)
        s = torch.sin(alpha / 2)

        matrix = torch.tensor([
            [c, 0, 0, -s],
            [0, c, -s, 0],
            [0, s, c, 0],
            [s, 0, 0, c]
        ], dtype=torch.float32, device=self.alpha.device)

        return matrix

    def _compute_rz_matrix(self) -> torch.Tensor:
        """
        Rz(beta) 계산:
        [e^(-iβ/2), 0        ]
        [0,         e^(iβ/2) ]

        → 4x4 실수 행렬로 변환:
        [cos(β/2), -sin(β/2), 0,        0       ]
        [sin(β/2),  cos(β/2), 0,        0       ]
        [0,         0,        cos(β/2), sin(β/2)]
        [0,         0,       -sin(β/2), cos(β/2)]
        """
        beta = self.beta
        c = torch.cos(beta / 2)
        s = torch.sin(beta / 2)

        matrix = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, c, s],
            [0, 0, -s, c]
        ], dtype=torch.float32, device=self.beta.device)

        return matrix

    def _embed_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ry(phi) 임베딩:
        입력 데이터를 quantum state로 변환
        Ry(φ) = [cos(φ/2), -sin(φ/2)]
                [sin(φ/2),  cos(φ/2)]
        """
        phi = torch.arctan(x)
        c = torch.cos(phi / 2)
        s = torch.sin(phi / 2)

        # [Re(a), Im(a), Re(b), Im(b)] 형태로 변환
        embedded = torch.stack([
            c,  # Re(cos(φ/2))
            torch.zeros_like(c),  # Im(cos(φ/2))
            s,  # Re(sin(φ/2))
            torch.zeros_like(s)  # Im(sin(φ/2))
        ], dim=-1)

        return embedded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 시퀀스를 처리하고 최종 출력을 반환

        Args:
            x: 입력 텐서, shape (batch_size, seq_length, input_size)
        Returns:
            output: 텐서, shape (batch_size, hidden_size)
        """
        batch_size, seq_length, _ = x.shape
        hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []

        for t in range(seq_length):
            # 1. 입력 임베딩 (Ry)
            embedded = self._embed_input(x[:, t, 0])  # input_size가 1

            # 2. weight_ih 적용 (Rx)
            rx_matrix = self._compute_rx_matrix()
            ih = rx_matrix @ embedded.t()

            # 3. weight_hh 적용 (Rz)
            rz_matrix = self._compute_rz_matrix()
            hh = rz_matrix @ hidden.t()

            # 4. 새로운 hidden state 계산
            hidden = ih.t() + hh.t()

            # 5. Normalization (quantum state의 norm 1 특성 유지)
            hidden = hidden / torch.norm(hidden, dim=1, keepdim=True)

            outputs.append(hidden)

        # 모든 시점의 hidden states를 쌓기
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_length, hidden_size)

        # Z-measurement 모사
        # 마지막 시점의 hidden state 사용
        final_hidden = outputs[:, -1, :]  # (batch_size, hidden_size)
        # hidden = [Re(a), Im(a), Re(b), Im(b)]
        re_a, im_a, re_b, im_b = final_hidden[:, 0], final_hidden[:, 1], final_hidden[:, 2], final_hidden[:, 3]

        # 확률 계산
        prob_a = re_a**2 + im_a**2  # |a|^2
        prob_b = re_b**2 + im_b**2  # |b|^2

        # 확률 분포 반환
        z_measurement = prob_a - prob_b
        z_measurement = z_measurement.unsqueeze(dim=-1)

        return z_measurement

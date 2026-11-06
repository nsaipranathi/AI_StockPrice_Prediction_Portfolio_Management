# LSTM Deep Learning Architecture & Mathematical Specification

## 1. LSTM Cell Gating Equations
The Long Short-Term Memory cell regulates information flow through three mathematical gates:

1. **Forget Gate**: Decides what information to discard from the cell state.
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2. **Input Gate**: Decides which new values to update in the cell state.
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

3. **Cell State Update**: Combines previous cell state and new candidate values.
   $$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

4. **Output Gate**: Determines the hidden state output.
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t * \tanh(C_t)$$

---

## 2. Keras Implementation Details
- **Sequence Length**: 60 Trading Days
- **Number of Input Features**: 10 (Close, Open, High, Low, Volume, SMA_20, SMA_50, EMA_20, RSI_14, MACD)
- **Layer 1**: 64 Units, `return_sequences=True`
- **Dropout 1**: Rate = 0.20
- **Layer 2**: 32 Units, `return_sequences=False`
- **Dropout 2**: Rate = 0.20
- **Dense Layer 1**: 16 Units, Activation = ReLU
- **Output Layer**: 1 Unit (Predicted Close Price)
- **Loss Function**: Huber Loss ($\delta = 1.0$)
- **Optimizer**: Adam ($lr = 0.001$)

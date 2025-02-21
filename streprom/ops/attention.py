import tensorflow as tf

def scaled_dot_product_attention(Q, K, V, #key_masks,
                                 causality=False, 
                                 dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''
    Q:[N, T_q, d_k]
    K:[N, T_k, d_k]
    V:[N, T_k, d_v]
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]
        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
        # scale
        outputs /= d_k ** 0.5
        # key masking
        #outputs = mask(outputs, key_masks=key_masks, type="key")
        # causality or future blinding masking
        #if causality:
            #outputs = mask(outputs, type="future")
        # softmax
        outputs = tf.nn.softmax(outputs) # (N, T_q, T_k)
        #attention = tf.transpose(outputs, [0, 2, 1]) 
        #tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
        # # query masking
        # outputs = mask(outputs, Q, K, type="query")
        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        # weighted sum (context vectors)=Z
        outputs = tf.matmul(outputs, V)  # (N, T_q, T_k)*V(N, T_k, d_v)=(N, T_q, d_v)
    return outputs


def multihead_attention(queries, keys, values, #key_masks,
                        num_heads=8, 
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''
    queries:[N, T_q, d_model].
    keys:[N, T_k, d_model].
    values:[N, T_k, d_model].
    '''
    assert d_model % num_heads == 0
    d_model = queries.get_shape().as_list()[-1] #512
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True) # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True) # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True) # (N, T_k, d_model)
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, d_model)
        # Residual connection
        outputs += queries
        # Normalize
        outputs = ln(outputs)
    return outputs #(N, T_q, C)  


def ff(inputs, num_units=[], scope="positionwise_feedforward"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = ln(outputs)
    return outputs #[N, T, C]


def ln(inputs, epsilon = 1e-8, scope="ln"): 
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

def positional_encoding(inputs,
                        maxlen, #>=T
                        masking=True,
                        scope="positional_encoding"): 
    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])
        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)
        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        # masks
        # if masking:
        #     outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
        return tf.to_float(outputs)


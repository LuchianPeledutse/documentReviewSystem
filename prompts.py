# IMPLEMENTATIONS OF PROMPTS 


def get_technologies_prompt(text: str):
    the_prompt = f"""
    Для приведенного ниже текста вычлени и передай в качестве ответа все 
    технологии в нужном формате 
    ТЕКСТ: {text}
    ФОРМАТ: технология_1, технология_2, технология_3, ..., технология_N
    ПРИМЕРЫ:
    ---
    Текст: Now let’s create the actual neural network. The placeholder X will act as the input
    layer; during the execution phase, it will be replaced with one training batch at a time
    (note that all the instances in a training batch will be processed simultaneously by the
    neural network). Now you need to create the two hidden layers and the output layer.
    The two hidden layers are almost identical: they differ only by the inputs they are
    connected to and by the number of neurons they contain. The output layer is also
    very similar, but it uses a softmax activation function instead of a ReLU activation
    function. So let’s create a neuron_layer() function that we will use to create one layer
    at a time. It will need parameters to specify the inputs, the number of neurons, the
    activation function, and the name of the layer
    Ответ: neural network, activation function
    ---
    Текст: The policy can be any algorithm you can think of, and it does not even have to be
    deterministic. For example, consider a robotic vacuum cleaner whose reward is the
    amount of dust it picks up in 30 minutes. Its policy could be to move forward with
    some probability p every second, or randomly rotate left or right with probability 1 
    p. The rotation angle would be a random angle between –r and +r. Since this policy
    involves some randomness, it is called a stochastic policy. The robot will have an
    erratic trajectory, which guarantees that it will eventually get to any place it can reach
    and pick up all the dust. The question is: how much dust will it pick up in 30
    minutes?
    How would you train such a robot? There are just two policy parameters you can
    tweak: the probability p and the angle range r. One possible learning algorithm could
    be  to try out many different values for these parameters, and pick the combination
    that performs best (see Figure 16-3). This is an example of policy search, in this case
    using a brute force approach. However, when the policy space is too large (which is
    generally the case), finding a good set of parameters this way is like searching for a
    needle in a gigantic haystack.
    Another way to explore the policy space is to use genetic algorithms. For example, you
    could randomly create a first generation of 100 policies and try them out, then “kill”
    Ответ: policy, probability, stochastic policy, learning algorithm
    ---
""" 
    return the_prompt


def get_suggestions_prompt(main_technologies: str, retrieved_docs: str) -> str:
    """Based on retrieved docs from vector database returns suggestions"""
    the_prompt = f"""
    Задание: Даны основные технологии и дополнительная информация. На основе дополнительной 
    информации выведи 10 неповторяющихся технологии на русском языке которые можно добавить к основному тексту.
    Основные технологии: {main_technologies}
    Дополнительная информация: {retrieved_docs}
    ФОРМАТ ВЫВОДА: технология_1, технология_2, технология_3, ..., технология_N
    ВАЖНО: Кроме самих технологий (и контекста в котором они упоминаются) ничего не дописывать
    ПРИМЕРЫ:
    ---
    Основные технологии: scikit-learn, XGBoost, TensorFlow, PyTorch
    Дополнительная информация: We’ve seen that we get almost perfect recall when our sparse retriever returns 
    documents, but can we do better at smaller values of k? The advantage of doing so is
    that we can pass fewer documents to the reader and thereby reduce the overall
    latency of our QA pipeline. A well-known limitation of sparse retrievers like BM25 is
    that they can fail to capture the relevant documents if the user query contains terms
    that don’t match exactly those of the review. One promising alternative is to use dense
    embeddings to represent the question and document, and the current state of the art
    is an architecture known as Dense Passage Retrieval (DPR).14 The main idea behind
    DPR is to use two BERT models as encoders for the question and the passage. As
    illustrated in Figure 7-10, these encoders map the input text into a d-dimensional
    k = 10 vector representation of the [CLS] token
    Ответ: recall, sparse retriever, QA pipeline, BM25, DPR
    ---
    """
    return the_prompt


"--------------------------------------------------------------------------------"
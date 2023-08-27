# Deep-Q-Learning-and-RL-GANs
In this repository, I intend to implement Deep Q-Learning for the game Connect4. Additionally, in the initial section, I will provide explanations about the attached files regarding RL-GAN networks.


In this project, we intend to become familiar with some applications of Reinforcement Learning in two sections.

In the first section, after getting acquainted with GANs, we aim to investigate the combination of GANs with Reinforcement Learning. We will examine RL-GAN networks and their application in completing images with cloud-like noise. In this section, we will utilize the following article and explain its algorithm:

"RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion" ([Link](https://arxiv.org/pdf/1904.12304.pdf))

In the second section, we plan to implement the game "Connect4" using Deep Q-Learning. In this part, we will begin by setting up the environment using the "kaggle-environment" library and then employ the Deep Q-Learning algorithm to train an agent. To better understand the game environment and algorithm implementation, the following resources have been used:

  • https://www.kaggle.com/code/ajeffries/connectx-getting-started/notebook

  • https://www.kaggle.com/code/gordotron85/teaching-an-agent-to-play-connect-4

  • https://medium.com/@louisdhulst/training-a-deep-q-learning-network-for-connect-4-9694e56cb806

**Part 1: Introduction to RL-GANs**
**Introduction to GANs:** GANs consist of two models, a Generator and a Discriminator, each constantly interacting with the other. The Generator attempts to approximate the probability distribution of the input and a noisy input vector, producing an output that closely resembles real data. On the other hand, the Discriminator aims to distinguish between real (original) data and generated (synthetic) data. These two components are constantly in interaction, enhancing each other. In fact, the Generator network is used to train the Discriminator network, while the Generator generates data for the Discriminator. As a result, this process is a form of Unsupervised Learning. One of the applications of the Discriminator network is detecting whether banknotes are genuine or counterfeit.

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/8949024c-5c53-4469-a027-bf67b4f96cbe)

Each of the sections intends to minimize or maximize the objective function of the problem in contrast to the other. As a result, these two components constantly drive each other towards progress until they ultimately reach a point of equilibrium (stability). The generator network has a loss so that if its output doesn't resemble the original data sufficiently, it gets penalized. On the other side, the discriminator also has a loss function to penalize it if it fails to distinguish real data from the synthetic one. Consequently, based on the illustration below, the objective function of GAN is defined as follows, where we are dealing with a min-max optimization problem.

Now, we want to become familiar with the algorithm and operation of RL-GANs in the provided article.

In many applications, when dealing with three-dimensional data, we use them in the form of point clouds. Often, when these data are in the form of point clouds, they have limitations and may be incomplete. In this article, a method has been designed using the combination of reinforcement learning and GANs to fill these empty regions in the best possible way. Essentially, in this method, a learning agent controls, to some extent, the ability of a GAN network to predict incomplete data and create their complete forms.

The impact of the RL-Agent in the GAN architecture is that it doesn't generate input noise for the Generator network randomly; instead, it's intelligently produced within the RL-GAN network. The image below illustrates the architecture of the RL-Agent network for completing inputs (filling in points). (Training GAN networks on GFV, which are feature vectors, improves the learning process.)

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/0db764f9-d8a1-49d8-b4b7-2b43ff40faf3)

The main structure of the RL-GAN method consists of three components:

1- Auto Encoder
2- Latent Space GAN
3- RL Agent

Each of these three components is essentially a neural network that requires separate training. First, we train the Auto Encoder (AE), and with the encoded outputs from the AE, we train the l-GAN networks. The RL agent also learns by interacting with these two networks.

The image below depicts the Auto Encoder component. As seen in the previous image as well, the goal is to minimize the error between the output and input in this section.

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/07a33a0c-63b9-49ba-b64d-6f23954f76bc)

The following image illustrates the latent space of a GAN (Generative Adversarial Network). As mentioned in the initial explanations, this section depicts the structure of a GAN network used for training two networks: the Generator and the Discriminator.

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/e3b21bde-55b1-47fd-95e6-7eb4c4e3e927)

In the above structure, the input of the Generator network, denoted as 'z', needs to be determined using an RL-Agent, the structure of which is also provided in the following image.

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/d006618c-d540-4d91-98a7-b914494194e3)

In the learning process, three loss functions have been considered for the three components: Discriminator, Auto Encoder, and Generator output. First, we define the Chamfer Distance criterion as follows:

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/753d1c27-9d3d-419b-a16f-d81878e83f6a)

We define the loss function of each component as follows:

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/15db9bce-d0ce-4046-af5a-51a6c124d123)

Now, using the aforementioned loss functions, we also define the reward component of the Reinforcement Learning (RL) as follows (where ωs are the weights of each loss function).

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/ecb36848-3ca7-4458-80a4-cd07de128567)

Finally, we model the training using RL-GAN-net as depicted in the following diagram. As observed in the image, the overall Reward is obtained by utilizing the output rewards of each network. Additionally, the input state of the RL-Agent is obtained by combining the output state (essentially the output of the Generator) and the output of the Auto Encoder.

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/6ecec60f-c0fe-487f-8ed3-9dd4582aa6a2)

The algorithm related to the implementation of RL-GAN will also be as depicted in the following image.

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/8263bdf9-7de1-450c-9063-689b3aacb25e)

As can also be seen in the above illustration, the inputs to the RL-Agent are the total reward and the state. We discussed the total reward. The input for the state is also the encoded initial image (P_in). Furthermore, the output of the RL-Agent is the input vector to the Generator network. To obtain policies as well as Q-values at each stage of the Deep Deterministic Policy Gradient (DDPG) algorithm, we employ it. This algorithm encompasses two networks: the actor and the critic. The critic network is updated using the Bellman equation, and the actor network is penalized (updated) using the gradient of the cost function, which is as follows (also referred to as the policy gradient).

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/4aaa723f-3f44-471d-9813-228b3a04381a)

A notable point about the network architecture is that the Generator and Discriminator networks have been pre-trained. The output of the Auto Encoder is also indicated by $P_out$. In the learning process (similar to Deep Q-learning), the networks are updated using a stored memory for actions, states, and rewards. The updating process occurs each time an input, $P_in$, is received. Upon receiving the input, $P_in$, initially, the RL-Agent makes a random move, and then using the noisy feature space, $GFV_n$, it selects an action ($a_t$). Finally, the decoder decodes the noiseless feature space, $GFV_c$, to produce the filled-in image (added points) as output. All of these components are also observable in the overall image.

With the explanation of the above algorithm, we can describe the general function of RL-GAN-net more simply as follows:
"The Generator network effectively acts as an Agent in an RL problem, attempting to maximize the received reward. Additionally, the Discriminator network returns the reward to the Agent at each stage."

Model performance evaluation: The image below illustrates the comparison of model parameters for different percentages of lost points. As observed, with the reduction of points (increasing percentage of lost points), the accuracy of the model decreases for the initial image.

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/ac1115af-b48f-404a-8a8a-f372e8c693ad)

Furthermore, the following image also demonstrates a comparison of the model's output for a scenario where 70% of input points are lost.

![image](https://github.com/ErfanPanahi/Deep-Q-Learning-and-RL-GANs/assets/107314081/4cbaa2e1-a9ba-4d24-ba9d-726e1634c40c)


**Part 2: Implementation of the Connect4 game using Deep Q-Learning**

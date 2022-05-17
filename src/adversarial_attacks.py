import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class AdversarialAttack:
    def __init__(self, model, eps=None):
        """
        :param model: instance of tf.keras.Model that is used to generate adversarial examples with attack
        :param eps: float number - maximum perturbation size of adversarial attack
        """
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()  # Loss that is used for adversarial attack
        self.model = model      # Model that is used for generating the adversarial examples
        self.eps = eps          # Threat radius of adversarial attack
        self.specifics = None   # String that contains all hyperparameters of attack
        self.name = None        # Name of the attack - e.g. FGSM


class Fgsm(AdversarialAttack):
    def __init__(self, model, eps):
        """
        :param model: instance of tf.keras.Model that is used for generating adversarial examples
        :param eps: floate number = maximum perturbation size in adversarial attack
        """
        super().__init__(model, eps)
        self.name = "FGSM"
        self.specifics = "FGSM - eps: {:.2f}".format(eps)

    def __call__(self, clean_images, true_labels):
        """
        :param clean_images: tf.Tensor - shape (n,h,w,c) - clean images that will be transformed to adversarial examples
        :param true_labels: tf.Tensor shape (n,) - true labels of clean images
        :return: tf.Tensor - shape (n,h,w,c) - adversarial examples generated with FGSM Attack
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # Only gradient w.r.t clean_images is accumulated NOT w.r.t model parameters
            tape.watch(clean_images)
            prediction = self.model(clean_images)
            loss = self.loss_obj(true_labels, prediction)

        gradients = tape.gradient(loss, clean_images)
        perturbations = self.eps * tf.sign(gradients)

        adv_examples = clean_images + perturbations
        adv_examples = tf.clip_by_value(adv_examples, 0, 1)
        return adv_examples


class PgdRandomRestart(AdversarialAttack):
    def __init__(self, model, eps, alpha, num_iter, restarts):
        """
        :param model: instance of tf.keras.Model that is used to generate adversarial examples
        :param eps: float number - maximum perturbation size for adversarial attack
        :param alpha: float number - step size in adversarial attack
        :param num_iter: integer - number of iterations of pgd during one restart iteration
        :param restarts: integer - number of restarts
        """
        super().__init__(model, eps)
        self.name = "PGD With Random Restarts"
        self.specifics = "PGD With Random Restarts - " \
                         f"eps: {eps} - alpha: {alpha} - " \
                         f"num_iter: {num_iter} - restarts: {restarts}"
        self.alpha = alpha
        self.num_iter = num_iter
        self.restarts = restarts

    def __call__(self, clean_images, true_labels):
        """
        :param clean_images: tf.Tensor - shape (n,h,w,c) - clean images will be transformed into adversarial examples
        :param true_labels: tf.Tensor- shape (n,) - true labels of clean_images
        :return: adversarial examples generated with PGD with random restarts
        """
        # Get loss on clean_images
        max_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)(true_labels, self.model(clean_images))
        # max_X contains adversarial examples and is updated after each restart
        max_X = clean_images[:, :, :, :]

        # Start restart loop
        for i in tf.range(self.restarts):
            # Get random perturbation uniformly in l infinity epsilon ball
            random_delta = 2 * self.eps * tf.random.uniform(shape=clean_images.shape) - self.eps
            # Add random perturbation
            X = clean_images + random_delta

            # Start projective gradient descent from X
            for j in tf.range(self.num_iter):
                # Track gradients
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    # Only gradients w.r.t. X are taken NOT model parameters
                    tape.watch(X)
                    pred = self.model(X)
                    loss = self.loss_obj(true_labels, pred)

                # Get gradients of loss w.r.t X
                gradients = tape.gradient(loss, X)
                # Compute perturbation as step size times sign of gradients
                perturbation = self.alpha * tf.sign(gradients)
                # Update X by adding perturbation
                X = X + perturbation
                # Make sure X did not leave L infinity epsilon ball around clean_images
                X = tf.clip_by_value(X, clean_images - self.eps, clean_images + self.eps)
                # Make sure X has entries between 0 and 1
                X = tf.clip_by_value(X, 0, 1)

            # Get crossentroby loss for each image in X
            loss_vector = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE)(true_labels, self.model(X))

            # mask is 1D tensor where true values are the rows of images that have higher loss than previous restarts
            mask = tf.greater(loss_vector, max_loss)
            # Update max_loss
            max_loss = tf.where(mask, loss_vector, max_loss)
            """
            we cannot do max_X[mask] = X[mask] like in numpy. We need mask that fits shape of max_X.
            Keep in mind that we want to select the rows that are True in the 1D tensor mask.
            We can simply stack the mask along the dimensions of max_X to select each desired row later.
            """
            # Create 2D mask of shape (max_X.shape[0],max_X.shape[1])
            multi_mask = tf.stack(max_X.shape[1] * [mask], axis=-1)
            # Create 3D mask of shape (max_X.shape[0],max_X.shape[1], max_X.shape[2])
            multi_mask = tf.stack(max_X.shape[2] * [multi_mask], axis=-1)
            # Create 4D mask of shape (max_X.shape[0],max_X.shape[1], max_X.shape[2], max_X.shape[3])
            multi_mask = tf.stack(max_X.shape[3] * [multi_mask], axis=-1)

            # Replace adversarial examples max_X[i] that have smaller loss than X[i] with X[i]
            max_X = tf.where(multi_mask, X, max_X)

        # return adversarial examples
        return max_X

class DeepFool(AdversarialAttack):

    def __init__(self, model, num_iter):
        """
        :param model: instance of tf.keras.Model that is used to generate adversarial examples
        :param eps: float number - maximum perturbation size for adversarial attack
        :param alpha: float number - step size in adversarial attack
        :param num_iter: integer - number of iterations (Recommended : 100)
        :param restarts: integer - number of restarts
        """
        super().__init__(model)
        self.name = "DeepFool"
        self.specifics = "DeepFool - " \
                         f"num_iter: {num_iter}"
        self.num_iter = num_iter
        self.loss_obj = tf.keras.losses.CategoricalCrossentropy()

    def fool(self, x, y, overshoot=1e-3):

        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.int32)

        attacked = False
        n_iter = 0
        x0 = x  # Initialize x0 = x, i = 0
        xs = [x0]

        sorted_indices = tf.squeeze(tf.argsort(self.model(tf.expand_dims(x, axis=0))))[::-1]  # The original label is at index 0, the others are 1-9


        def _cond(n_iter, xi, r_tot):
          predicted_label = tf.argmax(tf.squeeze(self.model(tf.expand_dims(xi, axis=0))))
          predicted_label = tf.cast(predicted_label, "int32")
          return tf.logical_and(tf.equal(predicted_label, tf.reduce_sum(y)),
                               n_iter < self.num_iter)

        def _body(n_iter, xi, r_tot):

            pert = np.inf
            w = tf.zeros_like(x)

            with tf.GradientTape() as tape:
                tape.watch(xi)
                prediction = tf.squeeze(self.model(tf.expand_dims(xi, axis=0)))
                y_onehot = tf.one_hot(y, prediction.shape[-1])
                loss = self.loss_obj(tf.squeeze(y_onehot), prediction)
            
            original_gradient = tape.gradient(loss, xi)  # Compute original gradient

            for k in range(1, prediction.shape[-1]): # Line 6 : for k != k0 do
                onehot_label = tf.one_hot(sorted_indices[k], prediction.shape[-1])
                
                with tf.GradientTape() as tape:
                    tape.watch(xi)
                    predi = self.model(tf.expand_dims(xi, axis=0))
                    partial_loss = self.loss_obj(onehot_label, tf.squeeze(predi))
                
                partial_gradient = tape.gradient(partial_loss, xi)
                w_k = partial_gradient - original_gradient  # Line 7
                f_k = prediction[sorted_indices[k]] - prediction[sorted_indices[0]]  # Line 8

                pert_k = abs(f_k) / tf.norm(tf.reshape(w_k, [-1]))

                if pert_k < pert:  # Eventually finds the argmin
                    pert = pert_k
                    w = w_k
                
            
            ri = -((pert + 1e-6) * w) / tf.norm(w)
            r_tot = r_tot + ri
            xi = xi + (1+overshoot)*ri
            xi = tf.clip_by_value(xi, 0, 1)

            xs.append(xi)

            # Find if the example has been sufficiently attacked
            predicted_label = tf.argmax(tf.squeeze(self.model(tf.expand_dims(xi, axis=0))))
            predicted_label = tf.cast(predicted_label, "int32")

            return n_iter+1, xi, r_tot

        _, xi, r_tot = tf.while_loop(_cond, _body, [0, x0, tf.zeros_like(x0)])
            
        r_tot = (1 + overshoot) * r_tot
        return xi, r_tot

    def __call__(self, clean_images, true_labels):
        """
        :param clean_images: tf.Tensor - shape (n,h,w,c) - clean images will be transformed into adversarial examples
        :param true_labels: tf.Tensor- shape (n,) - true labels of clean_images
        :return: adversarial examples generated with PGD with random restarts
        """

        if isinstance(clean_images, tf.Tensor):
          # print("Tensor")
          X_attack_list = []
          for i in range(clean_images.get_shape().as_list()[0]):
            x = clean_images[i, :, :, :]
            y = true_labels[i]
            # print(f"Attacking image {i}/{len(true_labels)}...")
            x_attack, _ = self.fool(x, y)
            X_attack_list.append(x_attack)
          X_attack = tf.stack(X_attack_list)
        else:
          X_attack = np.zeros(clean_images.shape)
          for i, (x, y) in enumerate(zip(clean_images, true_labels)):
              # print(f"Attacking image {i}/{len(true_labels)}...")
              x_attack, _ = self.fool(x, y)
              X_attack[i] = x_attack[-1]
        return X_attack


def attack_visual_demo(model, Attack, attack_kwargs, images, labels):
    """ Demo of adversarial attack on 20 images, visualizes adversarial robustness on 20 images
    :param model: tf,keras.Model
    :param Attack: type attacks.AdversarialAttack
    :param attack_kwargs: dicitonary - keyword arguments to call of instance of Attack
    :param images: tf.Tensor - shape (20, h, w, c)
    :param labels: tf.Tensor - shape (20,)
    :return Nothing
    """
    assert images.shape[0] == 20

    attack = Attack(model=model, **attack_kwargs)

    fig, axs = plt.subplots(4, 11, figsize=(15, 8))

    # Plot model predictions on clean images
    for i in range(4):
        for j in range(5):
            image = images[5 * i + j]
            label = labels[5 * i + j]
            ax = axs[i, j]
            ax.imshow(tf.squeeze(image), cmap="gray")
            ax.axis("off")

            prediction = model(tf.expand_dims(image, axis=0))
            prediction = tf.math.argmax(prediction, axis=1)
            prediction = tf.squeeze(prediction)
            color = "green" if prediction.numpy() == label.numpy() else "red"

            ax.set_title("Pred: " + str(prediction.numpy()),
                         color=color, fontsize=18)
    # Plot empty column
    for i in range(4):
        axs[i, 5].axis("off")

    # Set attack inputs
    if attack.name in ["Iterative Least Likely (Iter 1.1)",
                       "One Step Least Likely (Step 1.1)"]:
        attack_inputs = (images,)
    else:
        attack_inputs = (images, labels)

    # Get adversarial examples
    adv_examples = attack(*attack_inputs)

    # Plot model predictions on adversarial examples
    for i in range(4):
        for j in range(5):
            image = adv_examples[5 * i + j]
            label = labels[5 * i + j]
            ax = axs[i, 6 + j]
            ax.imshow(tf.squeeze(image), cmap="gray")
            ax.axis("off")

            prediction = model(tf.expand_dims(image, axis=0))
            prediction = tf.math.argmax(prediction, axis=1)
            prediction = tf.squeeze(prediction)
            color = "green" if prediction.numpy() == label.numpy() else "red"

            ax.set_title("Pred: " + str(prediction.numpy()),
                         color=color, fontsize=18)

    # Plot text
    plt.subplots_adjust(hspace=0.4)
    plt.figtext(0.16, 0.93, "Model Prediction on Clean Images", fontsize=18)
    plt.figtext(0.55, 0.93, "Model Prediction on Adversarial Examples", fontsize=18)
    plt.figtext(0.1, 1, "Adversarial Attack: "+attack.specifics, fontsize=24)

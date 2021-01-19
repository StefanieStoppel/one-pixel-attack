import argparse
import logging
import os

import imageio
import numpy as np
import yaml
from IPython import embed

from common import get_image_array, get_probability_for_class, get_perturbed_images, setup_logging
from differential_evolution import init_population, gen_children
from models.base import get_model_from_name

CONFIG = None
_logger = logging.getLogger(__name__)


def fitness_function(prediction, true_class=None):
    """
    For non-targeted attacks, the fitness function is the negative probability of true class
    """
    return -get_probability_for_class(prediction, true_class)


def get_fit_population(fathers, children, fathers_predictions, children_predictions, true_class):
    final_population = list()
    for i in range(len(fathers_predictions)):
        father_fitness = fitness_function(fathers_predictions[i], true_class)
        child_fitness = fitness_function(children_predictions[i], true_class)
        if father_fitness < child_fitness:
            final_population.append(children[i])
        else:
            final_population.append(fathers[i])
    return np.array(final_population)


def find_adversary_image(image, model):
    original_predictions = model.predict(np.copy(image))
    true_label = original_predictions[0][0][1]
    true_label_probability = original_predictions[0][0][2]
    _logger.info("True label: {}, Probability: {}".format(true_label, true_label_probability))
    os.makedirs('output', exist_ok=True)
    imageio.imwrite('output/original.jpg', image[0])

    population = init_population(CONFIG)
    for i in range(CONFIG["num_iterations"]):
        _logger.debug("Iteration: {}".format(i))
        perturbed_images = get_perturbed_images(image, population)
        perturbed_predictions = model.predict(np.copy(perturbed_images), top=model.num_classes)

        true_class_probabilities = [get_probability_for_class(p, true_label) for p in perturbed_predictions]
        min_prob = min(true_class_probabilities)
        max_prob = max(true_class_probabilities)
        _logger.debug("Probabilites for true class: Min={}, Max={}".format(min_prob, max_prob))
        if i % 10 == 0:
            write_image = perturbed_images[true_class_probabilities.index(min_prob)]
            imageio.imwrite('output/{}.jpg'.format(i), write_image.astype(np.uint8))
            _logger.info(f"Min probability for true class '{true_label}' for output image '{i}.jpg': {min_prob}.")

        population_children = gen_children(population, CONFIG)
        perturbed_images_children = get_perturbed_images(image, population_children)
        perturbed_predictions_children = model.predict(np.copy(perturbed_images_children), top=model.num_classes)

        population = get_fit_population(population,
                                        population_children,
                                        perturbed_predictions,
                                        perturbed_predictions_children,
                                        true_class=true_label)
    embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_file', help='config file')
    parser.add_argument('--input', '-i', dest='input_image', help='input image file')
    parser.add_argument('--loglevel', '-l', dest='log_level',
                        help='set log level, default: logging.INFO', default=logging.INFO)
    args = parser.parse_args()

    setup_logging(args.log_level)

    CONFIG = yaml.safe_load(open(args.config_file))
    model = get_model_from_name(CONFIG["model"])
    CONFIG["img_x"], CONFIG["img_y"], CONFIG["img_channels"] = model.input_size
    image_arr = get_image_array(args.input_image, config=CONFIG)
    find_adversary_image(image_arr, model)

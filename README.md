<h2>Overview</h2>
<p>This project provides a framework for training, testing, and evaluating deep learning models, including a Custom CNN and popular pre-trained models such as ResNet, GoogLeNet, EfficientNet, and DenseNet. It includes functionality for preprocessing images, cropping bounding boxes, and dynamically creating training and testing datasets from image and annotation files.</p>

<h2>Dependencies</h2>
<p>Ensure you have the following Python libraries installed:</p>
<ul>
    <li>torch</li>
    <li>torchvision</li>
    <li>Pillow</li>
    <li>tqdm</li>
</ul>

<h2>Usage</h2>

<h3>1. Create Training and Testing Sets</h3>
<p>Run the program with option <code>1</code> to generate training and testing datasets from image files and their corresponding XML annotations.</p>
<pre><code>python script.py</code></pre>
<p>Select option <code>1</code> and provide the required folder paths and split files.</p>

<h3>2. Train or Test a Model</h3>
<p>Run the program with option <code>2</code> to train a new model or test an existing one:</p>
<pre><code>python script.py</code></pre>
<p>Select option <code>2</code> and follow the prompts:</p>
<ul>
    <li>Choose an existing model to load or train a new one.</li>
    <li>Specify the number of epochs for training.</li>
    <li>Optionally save the trained model to a file.</li>
    <li>Test the model on the test dataset to evaluate its accuracy.</li>
</ul>

<h2>CustomCNN Model</h2>
<p>The <code>CustomCNN</code> class is a Convolutional Neural Network implemented from scratch. It includes:</p>
<ul>
    <li>Four convolutional layers with Batch Normalization and ReLU activation.</li>
    <li>Dropout for regularization.</li>
    <li>Fully connected layers for classification.</li>
</ul>

<h2>File Structure</h2>
<ul>
    <li><strong>JPEGImages/</strong>: Contains all image files.</li>
    <li><strong>XMLAnnotations/</strong>: Contains XML files with bounding box annotations.</li>
    <li><strong>ImageSplits/</strong>: Contains <code>train.txt</code> and <code>test.txt</code> files listing the images for training and testing, respectively.</li>
    <li><strong>train/</strong> and <strong>test/</strong>: Generated folders containing cropped images for training and testing.</li>
    <li><strong>models/</strong>: Directory for saving and loading trained model files.</li>
</ul>

<h2>Example Workflow</h2>
<ol>
    <li>Create the training and testing datasets by running the program and choosing option <code>1</code>.</li>
    <li>Train a new model or load an existing one by selecting option <code>2</code>.</li>
    <li>Test the model on the test dataset and evaluate its accuracy.</li>
    <li>Save the trained model if desired.</li>
</ol>

<h2>Notes</h2>
<ul>
    <li>The default input image size is 224x224.</li>
    <li>Pre-trained models use ImageNet weights.</li>
    <li>The framework supports dynamic cropping based on bounding box annotations in XML files.</li>
</ul>
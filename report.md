# REPORT

## WORKLOAD DIVISION
The workload was split equally between us two. We mostly worked together side by side during labs and at home. Some tasks that didn’t require important design choices were handled singularly (e.g. Julia handled implementing the audio loading and preprocessing, Elisa handled the main and the errors, Julia implemented docstrings, Elisa took care of type hints). However, for the most part the code was thought through and developed together.

## OVERVIEW OF THE CODE STRUCTURE AND INHERITANCES
The functionalities implemented by the code are displayed in the main. The classes can be divided in three main categories:
- a hierarchy of classes to manage datasets of audio an image files
- a batchloader that works with datasets to create batches of data for training a model
- a set of preprocessing techniques for image and audio data that can be applied in sequence by using a pipeline
Plus there is one class used only to implement very common errors that we raise multiple times in very similar ways throughout all of our other classes. 

We will now explain, for each of these three parts, what classes we have implemented and how these classes fit together through inheritance or composition. We will also explain the design choices for each class.

### DATASETS
The dataset handling is done by a total of 11 classes, and both inheritance and composition are used to coordinate them. The classes are the following:
- BaseDatasetABC(ABC) = an abstract class that serves as the conceptual base of all possible implementations of dataset classes.
- BaseDataset(BaseDatasetABC) = a class that inherits from the BaseDatasetABC and implements the logic that is shared among all dataset. This class takes care of both the case of having data with targets and the case of having unlabeled data, since these two cases can happen both in datasets handling images and in datasets handling audios, and also both in datasets made for regression and datasets made for classification. This class implements loading of the data in what we will refer to with “csv format” (having all files in a single folder and possibly a csv document associating each file with a target) because this format is shared among regression and classification datasets. It also implements what the subsetting operator returns, appropriate getters and setters for data and targets, and the train test split function. These functionalities are implemented at this level because they are shared by all types of datasets. What is not implemented instead is the handling of datasets that are hierarchically organized, since an organization of that type means that the dataset can only be used for classification, and the function that reads the files themselves, since these can be either images or audios (or potentially other types of files) so that function has to be implemented based on the specific dataset.
- ClassificationDataset(BaseDataset) = a class that inherits from BaseDataset and can be used to load datasets made for classification. It adds to it the possibility of loading datasets that are organized in hierarchical folders, with the folders being the different classes in which the data is organzied. By having this subclass be separated from the RegressionDataset class, other functionalities could be added in the future that only relate to datasets made for classification and would not be suited for datasets made for regression (for example implementing stratified sampling during train-test split to maintain the class distribution equal in both training and testing sets).
- RegressionDataset(BaseDataset) = a class that inherits from BaseDataset and implements the handling of datasets made for regression tasks. At the moment this class only implements an error that prevents the user from specifying “hierarchical” as a format, because hierarchically organized datasets cannot be datasets that are used for regression. However, as above, this class can serve as a base for implementing functionalities that only pertain to regression datasets (for example it could implement the loading of data in other formats only applicable in regression scenarios, or for example the scaling of the target variable, which only makes sense in regression datasets)
- ImageClassificationDataset(ClassificationDataset) = class that inherits from ClassificationDataset and makes it applicable to datasets composed of image files. It does this by crating an instance of the ImageLoader class, a callable class that specifically reads image files, and using that callable object to overwrite the “_read_data_file” function that was left unimplemented in the BaseDataset (here is where the principle of composition is used)
- ImageRegressionDataset(RegressionDataset) = class that inherits from RegressionDataset and, exactly as above, makes it applicable to datasets composed of images by using the callable class ImageLoader
- AudioClassificationDataset(ClassificationDataset) = class that inherits from ClassificationDataset and makes it applicable to datasets composed of audio files. It does it, as above, by creating an instance of the AudioLoader class, a callable class that reads audio files, and using that callable object to overwrite the “_read_data_file” function
- ImageRegressionDataset(RegressionDataset) = class that inherits from RegressionDataset and, again as above, makes it applicable to datasets composed of images by using the callable class AudioLoader
- DataLoaderABC(ABC) = abstract class that defines the overall structure of a data loader callable class
- ImageLoader(DataLoaderABC) = concrete callable class that inherits from DataLoaderABC and implements the reading of one image file
- AudioLoader(DataLoaderABC) = concrete callable class that inherits from DataLoaderABC and implements the reading of one audio file

In conclusion, our code uses both inheritance and composition to create a hierarchy of classes that can be used to build a wide variety of functionalities. Even though for the moment some classes look empty, this way the code base provides a solid and flexible foundation that allows for future development and handling of different types of datasets. This very spread out structure would allow a possible future developer to expand functionalities for regression a classification datasets without the need to change the whole hierarchical structure, and the use of separate callable classes to read in different formats of data make it really easy to adapt this implementation to every kind of data files (e.g. videos).

The following are the specific design choices for each class:

**BaseDatasetABC**

The following abstract methods and properties are included in this class because we believe that all datasets should include them:
- load_data() = abstract method to load the data, public because it’s the user’s tool to load the data into the object
- train_test_split() = abstract method to split the data into train and test set, public because it can be called by the user to perform the split
- data() = a getter and a setter for the data to allow the user to see and change the data safely (a getter and a setter for the targets are not included in this abstract class because not all datasets have targets)

**BaseDataset**

Attributes:
- _data, _targets = these attributes are private to prevent unsafe handling by the user, but can be viewed and modified safely by the user throught the getters and setters implemented as decorators. The getter allows the user to inspect the data and the targets after loading, and the setters allow the user to potentially be able to modify only the data or only the targets instead of having to reload data and targets together.
- _root, _labels_path, _format, _strategy = these attributes are private to prevent direct access from outside the class. Direct access is discouraged to maintain control over their state. Moreover, the user does not need to view these attributes since they’re directly set by them and never modified by the class logic, so no setters or getters are needed.
- _errors: an instance of the Errors class is used for handling errors and type checks. It is kept as a private attribute to encapsulate error-handling logic within the class.

Methods:
- data and targets (getter and setter methods): these methods provide controlled access to the _data and _targets attributes, ensuring that modifications go through proper type checks. They are marked as public for external access.
- _read_data_file(), _csv_to_labels(), _csv_load_data(), _get_item_format_helper(): private methods because they are part of the internal logic of the class and should not be accessed directly from instances of this class. _read_data_file() is a method that is not implemented and should be implemented in the subclasses, so it will throw an error if called by this class 
- train_test_split(): public method because it’s designed for external use

**ClassificationDataset**

Most attributes and methods of this class are inherited from the BaseDataset class and not modified. The modified or added methods are: 
- load_data(): public method overriding the method in BaseDataset. It implements the possibility to load data when the files are in hierarchical folders while keeping everything implemented in the parent class. Public because it’s the method called by the user to load the data.
- _hierarchical_load_data(): private method to encapsulate the implementation details of loading data hierarchically, so that the user can load the data in the specified format using always the same function.

**RegressionDataset**

Again, most attributes and methods of this class are inherited from the BaseDataset. The modified methods are: 
- load_data(): public method used to load the data that inherits the logic from BaseDataset but implements a check before that to rule out the possibility oft a user indicating a hierarchical folder structure.

**ImageClassificationDataset, ImageRegressionDataset, AudioClassificationDataset, ImageRegressionDataset**

These four classes inherit all attributes and methods from their respective parent classes. The only notable thing is that the _read_data_file function of these classes is overridden by a callable object (either of ImageLoader or AudioLoader class).

**DataLoaderABC**

Methods:
- _read_data_file(): abstract method that must be implemented by concrete subclasses. It represents the logic for reading data from a file and is expected to return the loaded data.

Magic methods:
- __call__(): method that provides a convenient way to use an instance of the class as a function. Calls the abstract _read_data_file() method, allowing concrete subclasses to implement the loading logic.

**ImageLoader and AudioLoader**

Both these two classes have only one method implemented that overrides the abstract method of their parent class:
- _read_data_file() = the method handles potential errors, such as the file not being able to be opened or being in the wrong format, and raises the appropriate exceptions.

### BATCHLOADER
The batchloading is handled by one single class, plus an abstract class from which it inherits:
- BatchLoaderABC = abstract class that makes a blueprint of all classes that can be defined as batchloaders. For instance in future implementation multiple ways of loading data might be implemented. In this scenario it could be useful to have a structure to follow.
- Batchloader = class constructed on top of the Dataset, which allows to load the data in batches. For each batch only the indices of the data points are stored and loaded into the program by using iterators.

The following are the specific design choices for each class:

**Batchloader**

Attributes:
- _batch_size, _dataset, _batches = all these private attributes are defined by the user when performing  the batchloading through the relative method. Moreover, the user does not need to view this attribute since it is directly set by them and never modified by the class logic, so no setters or getters are needed.
- _index = internal private attribute that keeps track of which batch has last been loaded. NO need for the user to know any of this, since it is solely used for the internal functioning of the class.
- _errors = an instance of the Errors class is used for handling errors and type checks. It is kept as a private attribute to encapsulate error-handling logic within the class.

Methods:
- create_batches = public method that the user can access to load data in batches.

**BatchLoaderABC**

Methods:
- create_batches:  abstract public attribute, it is necessary for all the classes that are defined as a batchloader.


### PREPROCESSING
There are 4 preprocessing techniques implemented:
- 2 for audio (pitchshifting and resampling)
- 2 for images (center cropping and random patching)

There are 5 classes in total: an abstract class, and one class for each preprocessing technique.
Finally, there is a data type agnostic class that alllows for multiple preprocessing methods to be applied in sequence (Pipeline).

- PreprocessingABC = an abstract class that serves as a blueprint for implementing any kind of preprocessing technique as a callable. In each subclass solely the _preprocessing_logic needs to be implemented where the processing steps are defined.
- PitchShifting = this class uses the functions from librosa to preprocess the audio data. The extent of the operation depends on the user’s input.
- Resampling = once again to carry out resampling some of the existing functionalities of librosa library where used. The new sampling rate of the signal depends on the user’s input.
- ImageCenterCrop = class that implements performs the center cropping of the image given height and the width (user’s input).
- ImagePatching = class carrying out the patching of an image given the color width and height (user’s input)
- Pipeline = given at least one callable it allows to carry out multiple reprocessing steps in sequence.

The following are the specific design choices for each class:

**PreprocessingABC**
- __ call __ = magic method defined in the abc class since each preprocessing step should be a callable object. Implementing this functionality in the abstract base class allows to have a general behviour across all the classes that need to carry out preprocessing.
- _preprocessing_logic = needs to be implemented n the concrete class.
- _preprocessing =  since image-with-target and audio-without-target have a similar structure (as shown in the table below), the code implements this private method to differentiate the actual data to be processed from the rest. This meethod is private since the user does not need to directly access it, since the functionalities of this method will be showcased anyway when the object is instantiated as a callable.

![image](https://github.com/rug-oop-2023/oop-final-project-group-7/assets/152076677/05f57b81-83f0-41d6-ab9b-42a79f8977e1)


**PitchShifting**

Attributes:
- pitch_shift = private attribute to prevent direct access to it from the user. Moreover, the user does not need to view this attribute since it is directly set by them and never modified by the class logic, so no setters or getters are needed.
- _errors =  an instance of the Errors class is used for handling errors and type checks. It is kept as a private attribute to encapsulate error-handling logic within the class.
  
Methods:
- _preprocessing_logic = imlementation of the preprocessing technique, namely the pitchshifting.

**Resampling**

Attributes:
- _new_sampling_rate = private attribute to define the new sampling rate of the signal. As mentioned before, the user does not need to view this attribute since it is directly set by them.
- _errors: an instance of the Errors class is used for handling errors and type checks. It is kept as a private attribute to encapsulate error-handling logic within the class.
  
Methods:
- _preprocessing_logic = imlementation of the preprocessing technique, namely the resampling.

**ImageCenterCropping**

Attributes:
- _width and _height = private attribute to define the width and height of the resulting cropped image. As mentioned before, the user does not need to view this attribute since it is directly set by them.
- _errors: an instance of the Errors class is used for handling errors and type checks. It is kept as a private attribute to encapsulate error-handling logic within the class.

Methods:
- _preprocessing_logic = imlementation of the preprocessing technique, namely the center cropping.

**ImagePatching**

Attributes:
- _color, _width and _height = private attributes to define the width, height and color of the patch. As mentioned before, the user does not need to view these attributes since it is directly set by them.
- _errors: an instance of the Errors class is used for handling errors and type checks. It is kept as a private attribute to encapsulate error-handling logic within the class.

Methods:
- _preprocessing_logic=imlementation of the preprocessing technique, namely the center cropping.

**Pipeline**

Attributes:
- _preprocessing_steps = private attribute(s) to define the preprocessing step(s) to be applied in series, it allows the class to be data-type agnostic. As mentioned before, the user does not need to view this attribute since it is directly set by them.

Methods:
- __ call __ =  allows the pipeline to be a callable (magic method)
- _apply_pipeline = private method that allows to call (all) the preprocessing step(s) defined in the attribute of the class.

### ERRORS
Class that has methods to carry out type-checks and value-checks of variables.

Methods:
- type_check = public method that can be used anytime that the user needs to carry out a type check.
- value_check = public method that can be used anytime that the user needs to carry out a value check.
- ispositive = public method that can be used anytime that the user needs to check for values to be positive.

All the other specific implementation details that were not explained were enforced by the assignment itself.

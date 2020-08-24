package org.sbml.spatial.segmentation;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import javax.imageio.ImageIO;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LearningKfold {

	private static final Logger log = LoggerFactory.getLogger(LearningKfold.class);
	private static final int WIDTH = 128;
	private static final int HEIGHT = 128;
	private static final int CHANNELS = 3;

	public static BufferedImage getBGRBufferedImage(BufferedImage bufferedImage) {
		for (int w = 0; w < bufferedImage.getWidth(); w++) {
			for (int h = 0; h < bufferedImage.getHeight(); h++) {
				int p = bufferedImage.getRGB(w, h);
				int a = (p >> 24) & 0xff;
				int r = (p >> 16) & 0xff;
				int g = (p >> 8) & 0xff;
				int b = p & 0xff;
				// swap r (red) and b (blue) channels
				p = (a << 24) | (b << 16) | (g << 8) | r;
				bufferedImage.setRGB(w, h, p);
			}
		}
		return bufferedImage;
	}

	public static BufferedImage getBGRBufferedImage(File imageFile) {
		try {
			BufferedImage bufferedImage = ImageIO.read(imageFile);
			return getBGRBufferedImage(bufferedImage);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	public static void main(String[] args) {
		try {

			int batchSize = 1;
			// String home = System.getProperty("user.home");

			String directory = System.getProperty("user.dir");
			String dataPath = directory + File.separator + "dataset";

			String pathToImage;
			if (args.length > 0) {
				pathToImage = args[0];
			} else {
				pathToImage = dataPath + File.separator + "raw_images" + File.separator + "F01_621w1_crop13.tif";
			}

			DataNormalization scaler = new ImagePreProcessingScaler(); // scale image between 0 and 1
			UnetPathLabelGenerator labeler = new UnetPathLabelGenerator();
			File rootDir = new File(dataPath + File.separator + "small_dataset");
			String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
			Random rng = new Random();
			FileSplit fileSplit = new FileSplit(rootDir, allowedExtensions, rng);
			BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labeler);

			int i, k = 4;
			double[] folds = new double[k];
			Arrays.fill(folds, 44 / k);
			InputSplit[] filesInDirSplit = fileSplit.sample(pathFilter, folds);

			ImageRecordReader[] imageReader = new ImageRecordReader[k];
			for (i = 0; i < k; i++) {
				imageReader[i] = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labeler);
				try {
					imageReader[i].initialize(filesInDirSplit[i]);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

			RecordReaderDataSetIterator[] set = new RecordReaderDataSetIterator[k];
			for (i = 0; i < k; i++) {
				set[i] = new RecordReaderDataSetIterator(imageReader[i], batchSize, 1, 1, true);
				scaler.fit(set[i]);
				set[i].setPreProcessor(scaler);

			}
			int numEpochs = 10;

			/*
			 * double splitTrainTest = 0.8;
			 * 
			 * InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 -
			 * splitTrainTest); InputSplit trainData = inputSplit[0]; InputSplit testData =
			 * inputSplit[1];
			 * 
			 * ImageRecordReader imageRecordReader = new
			 * ImageRecordReader(HEIGHT,WIDTH,CHANNELS,labeler);
			 * imageRecordReader.initialize(trainData); int labelIndex = 1;
			 */

			// KFoldIterator imageDataSetIterator = new
			// RecordReaderDataSetIterator(imageRecordReader,batchSize,labelIndex,labelIndex,true);
			// DataSetIterator imageDataSetIterator = new
			// RecordReaderDataSetIterator(imageRecordReader,batchSize,labelIndex,labelIndex,true);
			// DataSet imageDataSetIterator = (DataSet) new
			// RecordReaderDataSetIterator(imageRecordReader,batchSize,labelIndex,labelIndex,true);
			// DataSet imageData = (DataSet) imageRecordReader;
			// scaler.fit(imageData);
			// imageDataSetIterator.setPreProcessor(scaler);

			Map<Integer, Double> learningScheduleMap = new HashMap<>();
			learningScheduleMap.put(0, 0.00005);
			learningScheduleMap.put(200, 0.00001);
			learningScheduleMap.put(600, 0.000005);
			learningScheduleMap.put(800, 0.0000001);
//            learningScheduleMap.put(1000, 0.00001);

			int testFold = 0;
			while (testFold < k) {
				System.out.println("Testfold number:" + testFold + "   k value: " + k);

				ComputationGraph model = UNet.builder().updater(new Adam(1e-4)).build().init();
				model.addListeners(new ScoreIterationListener());

				System.out.println("Initializing new model");
				for (i = 0; i < k; i++) {
					if (i == testFold) {
						System.out.println("i number:" + i + "   k value: " + k);
						continue;
					} else {
						System.out.println("fitting model");
						model.fit(set[i], numEpochs);
					}
				}

				// Where to save the model
				// File locationTosave = new File(home + File.separator + "unetSave[" + testFold
				// + "]" + ".zip");
				File locationTosave = new File(directory + File.separator + "saveModel" + File.separator + "unetSave["
						+ testFold + "]" + ".zip");
				boolean saveUpdater = false;

				// ModelSerializer needs Model name, saveUpdater and Location of saving the
				// model

				ModelSerializer.writeModel(model, locationTosave, saveUpdater);
				testFold++;
			}

			log.info("*****EVALUATE MODEL******");
			NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
			BufferedImage image = ImageIO.read(new File(pathToImage));
			INDArray imageNative = loader.asMatrix(image);

			log.warn(imageNative.shapeInfoToString());

			imageNative = imageNative.reshape(1, CHANNELS, HEIGHT, WIDTH);
			imageNative = imageNative.divi(255f);

			/*
			 * INDArray[] output = model.output(imageNative); //INDArray sigmoid =
			 * Transforms.sigmoid(output); for (INDArray out : output) { out =
			 * out.reshape(1, HEIGHT,WIDTH); //out = out.permute(2,1,0); BufferedImage
			 * bufferedImage = new BufferedImage(WIDTH,HEIGHT,BufferedImage.TYPE_BYTE_GRAY);
			 * for ( i=0; i < WIDTH; i++) { for (int j=0; j < HEIGHT; j++) { float f =
			 * out.getFloat(new int[]{0,j,i}); int gray = (int)(f*255.0); // if (gray > 127)
			 * { // bufferedImage.setRGB(i,j,Color.WHITE.getRGB()); // } else { //
			 * bufferedImage.setRGB(i,j,Color.BLACK.getRGB()); // }
			 * bufferedImage.setRGB(i,j,new Color(gray,gray,gray).getRGB()); /*Color c = new
			 * Color(image.getRGB(j, i)); int red = (int)(c.getRed() * 0.299); int green =
			 * (int)(c.getGreen() * 0.587); int blue = (int)(c.getBlue() *0.114); Color
			 * newColor = new Color(red+green+blue, red+green+blue,red+green+blue);
			 * bufferedImage.setRGB(j,i,newColor.getRGB()); } }
			 * //ImageIO.write(bufferedImage,"tif",new File(home + File.separator +
			 * "outputUnet.tif")); // float[] values = out.toFloatVector(); //
			 * System.out.println(Arrays.toString(values)); }
			 */

		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}

	}

}

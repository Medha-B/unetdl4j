package org.sbml.spatial.segmentation;

import java.io.File;
import java.util.Random;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

/**
 * The class PreProcess.
 * 
 * This class is used for pre-processing the dataset to make it suitable for
 * fitting the U-Net model. Date Created: August 1, 2020
 * 
 * @author Medha Bhattacharya
 * @author Akira Funahashi
 * @author Yuta Tokuoka
 * @author Kaito Ii
 *
 */

public class PreProcess {
	/**
	 * int Width the width of input image set to 128
	 */
	public static final int WIDTH = 128;
	/**
	 * int HEIGHT the height of input image set to 128
	 */
	public static final int HEIGHT = 128;
	/**
	 * int Channels the number of input channels in image set to 3
	 */
	public static final int CHANNELS = 3;
	/**
	 * File RootDir the input file containing images
	 */
	private File Rootdir;
	/**
	 * int batchSize the size of every batch
	 */
	private int batchSize;

	/**
	 * Constructor for class PreProcess.java.
	 * 
	 * @param Rootdir   the input file (dataset)
	 * @param batchSize
	 */
	public PreProcess(File Rootdir, int batchSize) {
		this.Rootdir = Rootdir;
		this.batchSize = batchSize;
	}

	/**
	 * 
	 * Method for initializing the splitting of input imageRecordReader; handles
	 * exception.
	 * 
	 * @param imageRecordReader
	 * @param fileSplit
	 */

	public void Fsplit(ImageRecordReader imageRecordReader, FileSplit fileSplit) {
		try {
			imageRecordReader.initialize(fileSplit);
		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}

	}

	/**
	 * 
	 * The method that actually processes the input data and prepares it for model
	 * fitting.
	 * 
	 * @return imageDataSetIterator the DataSetIterator
	 */
	public DataSetIterator dataProcessed() {
		DataNormalization scaler = new ImagePreProcessingScaler();
		UnetPathLabelGenerator labeler = new UnetPathLabelGenerator();
		String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
		Random rng = new Random();
		FileSplit fileSplit = new FileSplit(Rootdir, allowedExtensions, rng);
		ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labeler);
		Fsplit(imageRecordReader, fileSplit);
		int labelIndex = 1;
		DataSetIterator imageDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize, labelIndex,
				labelIndex, true);
		scaler.fit(imageDataSetIterator);
		imageDataSetIterator.setPreProcessor(scaler);

		return imageDataSetIterator;
	}

}

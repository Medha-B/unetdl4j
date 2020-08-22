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

public class PreProcess {
	public static final int WIDTH = 128;
	public static final int HEIGHT = 128;
	public static final int CHANNELS = 3;
	private File Rootdir;
	private int batchSize;

	public PreProcess(File Rootdir, int batchSize) {
		this.Rootdir = Rootdir;
		this.batchSize = batchSize;
	}

	// For initializing the file split
	public void Fsplit(ImageRecordReader imageRecordReader, FileSplit fileSplit) {
		try {
			imageRecordReader.initialize(fileSplit);
		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}

	}

	// For data preparation
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

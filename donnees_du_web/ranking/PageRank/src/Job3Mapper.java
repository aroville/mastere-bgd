/***
 * Class Job3Mapper
 * Job3 Mapper class
 * @author sgarouachi
 */

import java.io.IOException;
import java.nio.charset.CharacterCodingException;
import java.util.Arrays;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class Job3Mapper extends Mapper<LongWritable, Text, FloatWritable, Text> {

	/**
	 * Job3 Map method
	 */
	@Override
	public void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String[] s = value.toString().split("\t");
        context.write(new FloatWritable(Float.parseFloat(s[1])), new Text(s[0]));
	}

	/**
	 * Get Page & Rank
	 * 
	 * @param key
	 * @param value
	 * @return String[2] String[0]: Page String[1]: Rank
	 * @throws CharacterCodingException
	 */
	private String[] getPageAndRank(LongWritable key, Text value)
			throws CharacterCodingException {
		// Get page and rank indexes
		String[] pageAndRank = new String[2];
		int tabPageIndex = value.find("\t");
		int tabRankIndex = value.find("\t", tabPageIndex + 1);

		// No tab after rank (when there are no links)
		int end;
		if (tabRankIndex == -1) {
			end = value.getLength() - (tabPageIndex + 1);
		} else {
			end = tabRankIndex - (tabPageIndex + 1);
		}

		// Get page & rank information
		pageAndRank[0] = Text.decode(value.getBytes(), 0, tabPageIndex);
		pageAndRank[1] = Text.decode(value.getBytes(), tabPageIndex + 1, end);

		// Return
		return pageAndRank;
	}

}

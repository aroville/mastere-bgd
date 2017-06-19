/***
 * Class Job2Mapper
 * Job2 Mapper class
 * @author sgarouachi
 */

import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class Job2Mapper extends Mapper<LongWritable, Text, Text, Text> {

	/**
	 * Job2 Map method Generates 3 outputs: Mark existing page: (pageI, !) Used
	 * to calculate the new rank (rank pageI depends on the rank of the inLink):
	 * (pageI, inLink \t rank \t totalLink) Original links of the page for the
	 * reduce output: (pageI, |pageJ,pageK...)
	 */
	@Override
	public void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String[] split = value.toString().split("\t");
		String page = split[0];
		Double rank = Double.parseDouble(split[1]);
		String[] links = split[2].split(",");

		context.write(new Text(page), new Text("!"));
		int n = links.length;
        Arrays.sort(links);

        for (String link: links)
            if (!link.isEmpty())
                 context.write(new Text(link), new Text(page+"\t"+rank+"\t"+n));

		context.write(new Text(page), new Text("|"+String.join(",", links)));
	}
}

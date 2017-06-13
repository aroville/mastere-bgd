/***
 * Class Job2Reducer
 * Job2 Reducer class
 * @author sgarouachi
 */

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public class Job2Reducer extends Reducer<Text, Text, Text, Text> {
    // Init dumping factor to 0.85
    private static final float damping = 0.85F;

    /**
     * Job2 Reduce method Calculate the new page rank
     */
    @Override
    public void reduce(Text page, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
        List<String> l = new ArrayList<>();
        String ts;
        boolean pageExists = false;
        String[] split;
        float sumPR=0;

        for (Text t : values) {
            ts = t.toString();

            if (ts.equals("!")) {
                pageExists = true;
                continue;
            }
            
            if (ts.startsWith("|")) {
                Collections.addAll(l, ts.substring(1).split(","));
                continue;
            }

            split = ts.split("\t");

            sumPR += Float.valueOf(split[1]) / Integer.valueOf(split[2]);
        }

        if (!pageExists)
            return;

        float newRank = 1-damping + damping*sumPR;
        for (String s: l)
            context.write(page, new Text(String.format(Locale.US,
                    "%.4f", newRank)+"\t"+s));
    }
}

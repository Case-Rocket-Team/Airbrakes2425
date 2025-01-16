float angleMapper(float inputAlt, float inputVel)
{
    //range of inputs expected: 6000-9200m 0-300m/s, lookup table has increments of 10m and 5m/s
    //make lookup tabke matrix
    float mapMatrix[320][60];
    
    //dummy matrix
    for (int i = 0; i < 320; i++)
    {
        for (int j = 0; j < 60; j++)
        {
            mapMatrix[i][j] = (100 * i) - (10 * j);
        }
    }
    
    //max value conditions, returns 90 deg airbrake angle if above 9200m or faster than 300m/s
    if (inputAlt > 9200)
    {
        return 90;
    }
    if (inputVel > 300)
    {
        return 90;
    }

    //convert altitude and speed readings to nearest increment in lookup table
    int n = (inputAlt - 6000) / 10;
    int m = inputVel / 5;

    //find the corresponding point in lookup table
    return mapMatrix[n][m];
}

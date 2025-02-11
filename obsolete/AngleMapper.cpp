float angleMapper(float inputAlt, float inputVel)
{
    //make dummy matrix 6000-9200m 0-300m/s
    float mapMatrix[320][60];
    
    for (int i = 0; i < 320; i++)
    {
        for (int j = 0; j < 60; j++)
        {
            mapMatrix[i][j] = (100 * i) - (10 * j);
        }
    }
    
    //max value conditions
    if (inputAlt > 9200)
    {
        return 90;
    }
    if (inputVel > 300)
    {
        return 90;
    }

    return mapMatrix[int((inputAlt - 6000) / 10)][int(inputVel / 5)];
}

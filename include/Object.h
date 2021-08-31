class Object{
    public:
        char* uid;
        double x,y;
        double width,height;

        static Object& generateFromFile(char* path);
};
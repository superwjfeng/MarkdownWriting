# 对象模型

一个class的封装的布局成本并不来自于类成员和类方法。类成员和struct一样，每个实例化对象保存一份。非内联函数则整个同一类的实例对象共享一份，内联函数则

主要开销在于virtual 函数



```c++
class Point {
public:
    Point(float xval);
    virtual ~Point();
    float x() const;
    static int PointCount();
protected:
    virtual ostream &print(ostream &os) const;
protected:
    float x_;
    static int point_count_;
}
```



# 构造函数语意学

# Data语意学

# Function语意学

# 构造、析构、拷贝语意学

# 执行期语意学

# 对象模型的顶端
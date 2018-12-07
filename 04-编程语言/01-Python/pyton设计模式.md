# Python 设计模式

- [python之路，Day24 常用设计模式学习](https://www.cnblogs.com/alex3714/articles/5760582.html)

## 1. 设计模式介绍

> **设计模式（Design Patterns）**
>
> ​					——可复用面向对象软件的基础

设计模式（Design pattern）是一套被反复使用、多数人知晓的、经过分类编目的、代码设计经验的总结。使用设计模式是为了可重用代码、让代码更容易被他人理解、保证代码可靠性。 毫无疑问，设计模式于己于他人于系统都是多赢的，设计模式使代码编制真正工程化，设计模式是软件工程的基石，如同大厦的一块块砖石一样。项目中合理的运用设计模式可以完美的解决很多问题，每种模式在现在中都有相应的原理来与之对应，每一个模式描述了一个在我们周围不断重复发生的问题，以及该问题的核心解决方案，这也是它能被广泛应用的原因。

## 2. 设计模式分类

经典的《设计模式》一书归纳出23种设计模式，这23种模式又可归为：

- 创建型
- 结构型
- 行为型

### 2.1 创建型模式

前面讲过，社会化的分工越来越细，自然在软件设计方面也是如此，因此对象的创建和对象的使用分开也就成为了必然趋势。因为对象的创建会消耗掉系统的很多资源，所以单独对对象的创建进行研究，从而能够高效地创建对象就是创建型模式要探讨的问题。这里有6个具体的创建型模式可供研究，它们分别是：

- 简单工厂模式（Simple Factory）

- 工厂方法模式（Factory Method）

- 抽象工厂模式（Abstract Factory）

- 创建者模式（Builder）

- 原型模式（Prototype）

- 单例模式（Singleton）

> 说明：严格来说，简单工厂模式不是GoF总结出来的23种设计模式之一。

### 2.2 结构型模式

在解决了对象的创建问题之后，对象的组成以及对象之间的依赖关系就成了开发人员关注的焦点，因为如何设计对象的结构、继承和依赖关系会影响到后续程序的维护性、代码的健壮性、耦合性等。对象结构的设计很容易体现出设计人员水平的高低，这里有7个具体的结构型模式可供研究，它们分别是：

- 外观模式（Facade）

- 适配器模式（Adapter）

- 代理模式（Proxy）

- 装饰模式（Decorator）

- 桥模式（Bridge）

- 组合模式（Composite）

- 享元模式（Flyweight）

### 2.3 行为型模式

在对象的结构和对象的创建问题都解决了之后，就剩下对象的行为问题了，如果对象的行为设计的好，那么对象的行为就会更清晰，它们之间的协作效率就会提高，这里有11个具体的行为型模式可供研究，它们分别是：

- 模板方法模式（Template Method）

- 观察者模式（Observer）

- 状态模式（State）

- 策略模式（Strategy）

- 职责链模式（Chain of Responsibility）

- 命令模式（Command）

- 访问者模式（Visitor）

- 调停者模式（Mediator）

- 备忘录模式（Memento）

- 迭代器模式（Iterator）

- 解释器模式（Interpreter）

## 3. 设计模式的六大原则

**1、开闭原则（Open Close Principle）**

开闭原则就是说**对扩展开放，对修改关闭**。在程序需要进行拓展的时候，不能去修改原有的代码，实现一个热插拔的效果。所以一句话概括就是：为了使程序的扩展性好，易于维护和升级。想要达到这样的效果，我们需要使用接口和抽象类，后面的具体设计中我们会提到这点。

**2、里氏代换原则（Liskov Substitution Principle）**

里氏代换原则(Liskov Substitution Principle LSP)面向对象设计的基本原则之一。 里氏代换原则中说，任何基类可以出现的地方，子类一定可以出现。 LSP是继承复用的基石，只有当衍生类可以替换掉基类，软件单位的功能不受到影响时，基类才能真正被复用，而衍生类也能够在基类的基础上增加新的行为。里氏代换原则是对“开-闭”原则的补充。实现“开-闭”原则的关键步骤就是抽象化。而基类与子类的继承关系就是抽象化的具体实现，所以里氏代换原则是对实现抽象化的具体步骤的规范。—— From Baidu 百科

**3、依赖倒转原则（Dependence Inversion Principle）**

这个是开闭原则的基础，具体内容：是对接口编程，依赖于抽象而不依赖于具体。

**4、接口隔离原则（Interface Segregation Principle）**

这个原则的意思是：使用多个隔离的接口，比使用单个接口要好。还是一个降低类之间的耦合度的意思，从这儿我们看出，其实设计模式就是一个软件的设计思想，从大型软件架构出发，为了升级和维护方便。所以上文中多次出现：降低依赖，降低耦合。

**5、迪米特法则（最少知道原则）（Demeter Principle）**

为什么叫最少知道原则，就是说：一个实体应当尽量少的与其他实体之间发生相互作用，使得系统功能模块相对独立。

**6、合成复用原则（Composite Reuse Principle）**

原则是尽量使用合成/聚合的方式，而不是使用继承。

## 工厂模式

工厂模式（Factory Pattern）是 Java 中最常用的设计模式之一。这种类型的设计模式属于创建型模式，它提供了一种创建对象的最佳方式。

在工厂模式中，我们在创建对象时不会对客户端暴露创建逻辑，并且是通过使用一个共同的接口来指向新创建的对象。

> **意图：**
>
> - 定义一个用于创建对象的接口，让子类决定实例化哪一个类。Factory Method 使**一**个类的实例化延迟到其子类。 
>
> **适用性：**
>
> - 当一个类不知道它所必须创建的对象的类的时候。
>
> - 当一个类希望由它的子类来指定它所创建的对象的时候。
>
> - 当类将创建对象的职责委托给多个子类中的某一个。

### 1. 简单工厂模式

<img src="_asset/设计模式-简单工厂模式.png">

<details><summary>简单工厂模式Code</summary>

```python
class ShapeFactory(object):
    '''工厂类'''
    def getShape(self):
      return self.shape_name

class Circle(ShapeFactory):
  def __init__(self):
    self.shape_name = "Circle"
  def draw(self):
    print('draw circle')

class Rectangle(ShapeFactory):
  def __init__(self):
    self.shape_name = "Retangle"
  def draw(self):
    print('draw Rectangle')

class Shape(object):
  '''接口类，负责决定创建哪个ShapeFactory的子类'''
  def create(self, shape):
    if shape == 'Circle':
      return Circle()
    elif shape == 'Rectangle':
      return Rectangle()
    else:
      return None

fac = Shape()
obj = fac.create('Circle')
obj.draw()
obj.getShape()
```

</details>

优点：客户端不需要修改代码。

缺点： 当需要增加新的运算类的时候，不仅需新加运算类，还要修改工厂类，违反了开闭原则。

### 2. 工厂方法模式

<img src="_asset/设计模式-工厂方法.png">

这个和简单工厂有区别，简单工厂模式只有一个工厂，工厂方法模式对每一个产品都有相应的工厂

好处：增加一个运算类（例如N次方类），只需要增加运算类和相对应的工厂，两个类，不需要修改工厂类。

缺点：增加运算类，会修改客户端代码，工厂方法只是把简单工厂的内部逻辑判断移到了客户端进行。

<details><summary>简单工厂模式Code</summary>

```python
class ShapeFactory(object):
    '''工厂类'''
    def getShape(self):
        return self.shape_name

class Circle(ShapeFactory):
    def __init__(self):
        self.shape_name = "Circle"
    def draw(self):
        print('draw circle')

class Rectangle(ShapeFactory):
    def __init__(self):
        self.shape_name = "Retangle"
    def draw(self):
        print('draw Rectangle')

class ShapeInterfaceFactory(object):
    '''接口基类'''
    def create(self):
        '''把要创建的工厂对象装配进来'''
        raise  NotImplementedError

class ShapeCircle(ShapeInterfaceFactory):
    def create(self):
        return Circle()

class ShapeRectangle(ShapeInterfaceFactory):
    def create(self):
        return Rectangle()

shape_interface = ShapeCircle()
obj = shape_interface.create()
obj.getShape()
obj.draw()

shape_interface2 = ShapeRectangle()
obj2 = shape_interface2.create()
obj2.draw()
```

</details>

下面这个是用学校＼课程来描述了一个工厂模式

<details><summary>简单工厂模式Code</summary>

```python
#_*_coding:utf-8_*_
__author__ = 'Mr.Miaow'
 
'''
工厂方法模式是简单工厂模式的衍生，解决了许多简单工厂模式的问题。
首先完全实现‘开－闭 原则’，实现了可扩展。其次更复杂的层次结构，可以应用于产品结果复杂的场合。 　　
工厂方法模式的对简单工厂模式进行了抽象。有一个抽象的Factory类（可以是抽象类和接口），这个类将不在负责具体的产品生产，而是只制定一些规范，具体的生产工作由其子类去完成。在这个模式中，工厂类和产品类往往可以依次对应。即一个抽象工厂对应一个抽象产品，一个具体工厂对应一个具体产品，这个具体的工厂就负责生产对应的产品。 　　
工厂方法模式(Factory Method pattern)是最典型的模板方法模式(Templete Method pattern)应用。
'''

class AbstractSchool(object):
    name = ''
    addr = ''
    principal = ''
 
    def enroll(self,name,course):
        pass
    def info(self):
        pass
    
class AbstractCourse(object):
    def __init__(self,name,time_range,study_type,fee):
        self.name = name
        self.time_range = time_range
        self.study_type = study_type
        self.fee = fee
 
    def enroll_test(self):
        '''
        参加这门课程前需要进行的测试
        :return:
        '''
        print("课程[%s]测试中..." % self.name) 
    def print_course_outline(self):
        '''打印课程大纲'''
        pass
    
class LinuxOPSCourse(AbstractCourse):
    '''
    运维课程
    '''
    def print_course_outline(self):
        outline='''
        Linux 基础
        Linux 基本服务使用
        Linux 高级服务篇
        Linux Shell编程
        '''
        print(outline)
    def enroll_test(self):
        print("不用测试,是个人就能学...")

class PythonCourse(AbstractCourse):
    '''python自动化开发课程'''
    def print_course_outline(self):
        outline='''
        python 介绍
        python 基础语法
        python 函数式编程
        python 面向对象
        python 网络编程
        python web开发基础
        '''
        print(outline)
    def enroll_test(self):
        print("-------python入学测试-------")
        print("-------500道题答完了-------")
        print("-------通过了-------")

class BJSchool(AbstractSchool):
    name = "老男孩北京校区"
    def create_course(self,course_type):
        if course_type == 'py_ops':
            course = PythonCourse("Python自动化开发",
                         7,'面授',11000)
        elif course_type == 'linux':
            course = LinuxOPSCourse("Linux运维课程",
                         5,'面授',12800)
        return course
 
    def enroll(self,name,course):
        print("开始为新学员[%s]办入学手续... "% name)
        print("帮学员[%s]注册课程[%s]..." % (name,course))
        course.enroll_test()
    def info(self):
        print("------[%s]-----"%self.name)
        
class SHSchool(AbstractSchool):
    name = "老男孩上海分校"
    def create_course(self,course_type):
        if course_type == 'py_ops':
            course = PythonCourse("Python自动化开发",
                         8,'在线',6500)
        elif course_type == 'linux':
            course = LinuxOPSCourse("Linux运维课程",
                         6,'在线',8000)
        return course
    def enroll(self,name,course):
        print("开始为新学员[%s]办入学手续... "% name)
        print("帮学员[%s]注册课程[%s]..." % (name,course))
        #course.level_test()
    def info(self):
        print("--------[%s]-----" % self.name )
        
school1 = BJSchool()
school2 = SHSchool()
 
school1.info()
c1=school1.create_course("py_ops")
school1.enroll("张三",c1)
 
school2.info()
c2=school1.create_course("py_ops")
school2.enroll("李四",c2)
```

</details>

### 3. 抽象工厂模式

每一个模式都是针对一定问题的解决方案。抽象工厂模式与工厂方法模式的最大区别就在于，工厂方法模式针对的是一个产品等级结构；而抽象工厂模式则需要面对多个产品等级结构。

在学习抽象工厂具体实例之前，应该明白两个重要的概念：产品族和产品等级。

所谓产品族，是指位于不同产品等级结构中，功能相关联的产品组成的家族。比如AMD的主板、芯片组、CPU组成一个家族，Intel的主板、芯片组、CPU组成一个家族。而这两个家族都来自于三个产品等级：主板、芯片组、CPU。一个等级结构是由相同的结构的产品组成，示意图如下：

<img src="_asset/设计模式-抽象工厂.png">

显然，每一个产品族中含有产品的数目，与产品等级结构的数目是相等的。产品的等级结构与产品族将产品按照不同方向划分，形成一个二维的坐标系。横轴表示产品的等级结构，纵轴表示产品族，上图共有两个产品族，分布于三个不同的产品等级结构中。只要指明一个产品所处的产品族以及它所属的等级结构，就可以唯一的确定这个产品。

上面所给出的三个不同的等级结构具有平行的结构。因此，如果采用工厂方法模式，就势必要使用三个独立的工厂等级结构来对付这三个产品等级结构。由于这三个产品等级结构的相似性，会导致三个平行的工厂等级结构。随着产品等级结构的数目的增加，工厂方法模式所给出的工厂等级结构的数目也会随之增加。如下图：

<img src="_asset/设计模式-抽象工厂-01.png">

那么，是否可以使用同一个工厂等级结构来对付这些相同或者极为相似的产品等级结构呢？当然可以的，而且这就是抽象工厂模式的好处。同一个工厂等级结构负责三个不同产品等级结构中的产品对象的创建。

<img src="_asset/设计模式-抽象工厂-02.png">

可以看出，一个工厂等级结构可以创建出分属于不同产品等级结构的一个产品族中的所有对象。显然，这时候抽象工厂模式比简单工厂模式、工厂方法模式更有效率。对应于每一个产品族都有一个具体工厂。而每一个具体工厂负责创建属于同一个产品族，但是分属于不同等级结构的产品。

#### 3.1 抽象工厂模式结构

抽象工厂模式是对象的创建模式，它是工厂方法模式的进一步推广。

假设一个子系统需要一些产品对象，而这些产品又属于一个以上的产品等级结构。那么为了将消费这些产品对象的责任和创建这些产品对象的责任分割开来，可以引进抽象工厂模式。这样的话，消费产品的一方不需要直接参与产品的创建工作，而只需要向一个公用的工厂接口请求所需要的产品。

通过使用抽象工厂模式，可以处理具有相同（或者相似）等级结构中的多个产品族中的产品对象的创建问题。如下图所示：

<img src="_asset/设计模式-抽象工厂-03.png">

由于这两个产品族的等级结构相同，因此使用同一个工厂族也可以处理这两个产品族的创建问题，这就是抽象工厂模式。

根据产品角色的结构图，就不难给出工厂角色的结构设计图。

<img src="_asset/设计模式-抽象工厂-04.png">

可以看出，每一个工厂角色都有两个工厂方法，分别负责创建分属不同产品等级结构的产品对象。

<img src="_asset/设计模式-抽象工厂-05.png">

抽象工厂的功能是为一系列相关对象或相互依赖的对象创建一个接口。一定要注意，这个接口内的方法不是任意堆砌的，而是一系列相关或相互依赖的方法。比如上面例子中的主板和CPU，都是为了组装一台电脑的相关对象。不同的装机方案，代表一种具体的电脑系列。

<img src="_asset/设计模式-抽象工厂-06.png">

由于抽象工厂定义的一系列对象通常是相关或相互依赖的，这些产品对象就构成了一个产品族，也就是抽象工厂定义了一个产品族。

这就带来非常大的灵活性，切换产品族的时候，只要提供不同的抽象工厂实现就可以了，也就是说现在是以一个产品族作为一个整体被切换。

<img src="_asset/设计模式-抽象工厂-07.png">

#### 3.2 在什么情况下应当使用抽象工厂模式

**1.一个系统不应当依赖于产品类实例如何被创建、组合和表达的细节，这对于所有形态的工厂模式都是重要的。**

**2.这个系统的产品有多于一个的产品族，而系统只消费其中某一族的产品。**

**3.同属于同一个产品族的产品是在一起使用的，这一约束必须在系统的设计中体现出来。**（比如：Intel主板必须使用Intel CPU、Intel芯片组）

**4.系统提供一个产品类的库，所有的产品以同样的接口出现，从而使客户端不依赖于实现。**

#### 3.3 抽象工厂模式的起源

抽象工厂模式的起源或者最早的应用，是用于创建分属于不同操作系统的视窗构建。比如：命令按键（Button）与文字框（Text)都是视窗构建，在UNIX操作系统的视窗环境和Windows操作系统的视窗环境中，这两个构建有不同的本地实现，它们的细节有所不同。

在每一个操作系统中，都有一个视窗构建组成的构建家族。在这里就是Button和Text组成的产品族。而每一个视窗构件都构成自己的等级结构，由一个抽象角色给出抽象的功能描述，而由具体子类给出不同操作系统下的具体实现。

#### 3.4 抽象工厂模式的优点

- **分离接口和实现**

  客户端使用抽象工厂来创建需要的对象，而客户端根本就不知道具体的实现是谁，客户端只是面向产品的接口编程而已。也就是说，客户端从具体的产品实现中解耦。

- **使切换产品族变得容易**

  因为一个具体的工厂实现代表的是一个产品族，比如上面例子的从Intel系列到AMD系列只需要切换一下具体工厂。

#### 3.5 抽象工厂模式的缺点

- **不太容易扩展新的产品**

  如果需要给整个产品族添加一个新的产品，那么就需要修改抽象工厂，这样就会导致修改所有的工厂实现类。






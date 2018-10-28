## Index

<!-- TOC -->

* [1. Python语言特性](#1-python语言特性)
  - [1.1 Python的函数参数传递](#11-python的函数参数传递)
  - [1. 2 Python中的元类(metaclass)](#1-2-python中的元类metaclass)
  - [1.3 @staticmethod 和 @classmethod](#13-staticmethod-和-classmethod)
  - [1.4 类变量和实例变量](#14-类变量和实例变量)
  - [1.5 Python自省](#15-python自省)
  - [1.6 字典推导式](#16-字典推导式)
  - [1.7 Python中单下划线和双下划线](#17-python中单下划线和双下划线)
  - [1.8 字符串格式化: \x 和 .format](#18-字符串格式化--和-format)
  - [1.9 迭代器和生成器](#19-迭代器和生成器)
  - [1.10 *args and <code>**kwargs</code>](#110-args-and-kwargs)
  - [1.11 面向切面编程 AOP 和装饰器](#111-面向切面编程-aop-和装饰器)
  - [1.12 鸭子类型](#112-鸭子类型)
  - [1.13 Python中重载](#113-python中重载)
  - [1.14 新式类和旧式类](#114-新式类和旧式类)
  - [1.15 __new__和<code>__init__</code>的区别](#115-__new__和__init__的区别)
  - [1.16 单例模式](#116-单例模式)
    - [1.16.1 使用__new__方法](#1161-使用__new__方法)
    - [1.16.2 共享属性](#1162-共享属性)
    - [1.16.3 装饰器版本](#1163-装饰器版本)
    - [1.16.4 import方法](#1164-import方法)
  - [1.17 Python 中的作用域](#117-python-中的作用域)
  - [1.18 GIL 线程全局锁](#118-gil-线程全局锁)
  - [1.19 协程](#119-协程)
  - [1.20 闭包](#120-闭包)
  - [1.21 lambda函数](#121-lambda函数)
  - [1.22 Python函数式编程](#122-python函数式编程)
  - [1.23 Python里的拷贝](#123-python里的拷贝)
  - [1.24 Python垃圾回收机制](#124-python垃圾回收机制)
    - [1.24.1 引用计数](#1241-引用计数)
    - [1.24.2 标记-清除机制](#1242-标记-清除机制)[1.24.3 分代技术](#1243-分代技术)
  - [1.25 Python的List](#125-python的list)
  - [1.26 Python的is](#126-python的is)
  - [1.27 read, readline 和 readlines](#127-read-readline-和-readlines)
  - [1.28 Python2 和 3 的区别](#128-python2-和-3-的区别)
* [2. 操作系统](#2-操作系统)
  - [2.1 select, poll 和 epoll](#21-select-poll-和-epoll)
  - [2.2 调度算法](#22-调度算法)
  - [2.3 死锁](#23-死锁)
  - [2.4 程序编译与链接](#24-程序编译与链接)
    - [2.4.1 预处理](#241-预处理)
    - [2.4.2 编译](#242-编译)
    - [2.4.3 汇编](#243-汇编)
    - [2.4.4 链接](#244-链接)
  - [2.5 静态链接和动态链接](#25-静态链接和动态链接)
  - [2.6 虚拟内存技术](#26-虚拟内存技术)
  - [2.7 分页和分段](#27-分页和分段)
    - [2.7.1 分页与分段的主要区别](#271-分页与分段的主要区别)
  - [2.8 页面置换算法](#28-页面置换算法)
  - [2.9 边沿触发和水平触发](#29-边沿触发和水平触发)
* [3. 数据库](#3-数据库)
  - [3.1 事务](#31-事务)
  - [3.2 数据库索引](#32-数据库索引)
  - [3.3 Redis原理](#33-redis原理)
  - [3.4 乐观锁和悲观锁](#34-乐观锁和悲观锁)
  - [3.5 MVCC](#35-mvcc)
  - [3.6 MyISAM 和 InnoDB](#36-myisam-和-innodb)
* [4. 网络](#4-网络)
  - [4.1 三次握手](#41-三次握手)
  - [4.2 四次挥手](#42-四次挥手)
  - [4.3 ARP协议](#43-arp协议)
  - [4.4 urllib 和 urllib2 的区别](#44-urllib-和-urllib2-的区别)
  - [4.5 Post 和 Get](#45-post-和-get)
  - [4.6 Cookie 和 Session](#46-cookie-和-session)
  - [4.7 apache和nginx的区别](#47-apache和nginx的区别)
  - [4.8 网站用户密码保存](#48-网站用户密码保存)
  - [4.9 HTTP和HTTPS](#49-http和https)
  - [4.10 XSRF 和 XSS](#410-xsrf-和-xss)
  - [4.11 幂等 Idempotence](#411-幂等-idempotence)
  - [4.12 RESTful架构(SOAP,RPC)](#412-restful架构soaprpc)
  - [4.13 SOAP](#413-soap)
  - [4.14 RPC](#414-rpc)
  - [4.15 CGI 和 WSGI](#415-cgi-和-wsgi)
  - [4.16 中间人攻击](#416-中间人攻击)
  - [4.17 c10k 问题](#417-c10k-问题)
  - [4.18 socket](#418-socket)
  - [4.19 浏览器缓存](#419-浏览器缓存)
  - [4.20 HTTP1.0和HTTP1.1](#420-http10和http11)
  - [4.21 Ajax](#421-ajax)
* [5. *NIX](#5-nix)
  - [5.1 unix进程间通信方式(IPC)](#51-unix进程间通信方式ipc)
* [6. 数据结构](#6-数据结构)
  - [6.1 红黑树](#61-红黑树)
* [7. 编程题](#7-编程题)
  - [7.1 台阶问题/斐波纳挈](#71-台阶问题斐波纳挈)
  - [7.2 变态台阶问题](#72-变态台阶问题)
  - [7.3 矩形覆盖](#73-矩形覆盖)
  - [7.4 杨氏矩阵查找](#74-杨氏矩阵查找)
  - [7.5 去除列表中的重复元素](#75-去除列表中的重复元素)
  - [7.6 链表成对调换](#76-链表成对调换)
  - [7.7 创建字典的方法](#77-创建字典的方法)
    - [7.7.1 直接创建](#771-直接创建)
    - [7.7.2 工厂方法](#772-工厂方法)
    - [7.7.3 fromkeys()方法](#773-fromkeys方法)
  - [7.8 合并两个有序列表](#78-合并两个有序列表)
  - [7.9 交叉链表求交点](#79-交叉链表求交点)
  - [7.10 二分查找](#710-二分查找)
  - [7.11 快排](#711-快排)
  - [7.12 找零问题](#712-找零问题)
  - [7.13 广度遍历和深度遍历二叉树](#713-广度遍历和深度遍历二叉树)
  - [7.14 前中后序遍历](#714-前中后序遍历)
  - [7.15 求最大树深](#715-求最大树深)
  - [7.16 求两棵树是否相同](#716-求两棵树是否相同)
  - [7.17 前序中序求后序](#717-前序中序求后序)
  - [7.18 单链表逆置](#718-单链表逆置)
  - [7.19 求两个数字之间的素数](#719-求两个数字之间的素数)
  - [7.20 请用Python手写实现冒泡排序](#720-请用python手写实现冒泡排序)
  - [7.21 请用Python手写实现选择排序](#721-请用python手写实现选择排序)
  - [7.22 请用Python手写实现插入排序](#722-请用python手写实现插入排序)
  - [7.23 请用Python手写实现快速排序](#723-请用python手写实现快速排序)
  - [7.24 请用Python手写实现堆排序](#724-请用python手写实现堆排序)
  - [7.25 请用Python手写实现归并排序](#725-请用python手写实现归并排序)
  - [7.26 链表操作](#726-链表操作)
* [Reference](#reference)

<!-- /TOC -->

## 1. Python语言特性

### 1.1 Python的函数参数传递

看两个例子:

```python
a = 1
def fun(a):
    a = 2
fun(a)
print a  # 1
```

```python
a = []
def fun(a):
    a.append(1)
fun(a)
print a  # [1]
```

**所有的变量都可以理解是内存中一个对象的 “引用”**，或者，也可以看似 c 中 `void*` 的感觉。

**这里记住的是类型是属于对象的，而不是变量**。而对象有两种, “可更改”（mutable）与 “不可更改”（immutable）对象。

在 python 中，**不可更改的对象**：

- strings
- tuples
- numbers

**可修改对象**：

- list
- dict

当一个引用传递给函数的时候, 函数自动复制一份引用, 这个函数里的引用和外边的引用没有半毛关系了. 所以第一个例子里函数把引用指向了一个不可变对象, 当函数返回的时候, 外面的引用没半毛感觉. 而第二个例子就不一样了,函数内的引用指向的是可变对象, 对它的操作就和定位了指针地址一样, 在内存里进行修改.

如果还不明白的话, [这里有更好的解释](http://stackoverflow.com/questions/986006/how-do-i-pass-a-variable-by-reference)

### 1. 2 Python中的元类(metaclass)

这个非常的不常用,但是像 ORM 这种复杂的结构还是会需要的, 详情请看：《[深刻理解Python中的元类(metaclass)](http://python.jobbole.com/21351/)》

### 1.3 @staticmethod 和 @classmethod

Python 其实有 3 个方法：

- 静态方法(staticmethod)
- 类方法(classmethod)
- 实例方法

```python
def foo(x):
    print "executing foo(%s)"%(x)
 
class A(object):
    def foo(self,x):
        print "executing foo(%s,%s)"%(self,x)
 
    @classmethod
    def class_foo(cls,x):
        print "executing class_foo(%s,%s)"%(cls,x)
 
    @staticmethod
    def static_foo(x):
        print "executing static_foo(%s)"%x
 
a=A()
```

这里先理解下函数参数里面的 self 和 cls。**这个 self 和 cls 是对类或者实例的绑定**，对于一般的函数来说我们可以这么调用 `foo(x)`，这个函数就是最常用的，它的工作跟任何东西(类, 实例)无关。

- 对于实例方法，我们知道在类里每次定义方法的时候都需要绑定这个实例，就是 `foo(self, x)`，为什么要这么做呢? **因为实例方法的调用离不开实例**，我们需要把实例自己传给函数，调用的时候是这样的 `a.foo(x)` (其实是`foo(a, x)`)。
- 类方法一样，**只不过它传递的是类而不是实例**，`A.class_foo(x)`。注意这里的 self 和 cls 可以替换别的参数，但是 python 的约定是这俩，还是不要改的好。

- 对于静态方法其实和普通的方法一样，不需要对谁进行绑定，唯一的区别是调用的时候需要使用 `a.static_foo(x)` 或者 `A.static_foo(x)` 来调用.

| \       | 实例方法 | 类方法         | 静态方法        |
| ------- | -------- | -------------- | --------------- |
| a = A() | a.foo(x) | a.class_foo(x) | a.static_foo(x) |
| A       | 不可用   | A.class_foo(x) | A.static_foo(x) |

[更多关于这个问题](http://stackoverflow.com/questions/136097/what-is-the-difference-between-staticmethod-and-classmethod-in-python)

### 1.4 类变量和实例变量

```python
class Person:
    name="aaa"
 
p1=Person()
p2=Person()
p1.name="bbb"
print p1.name  # bbb
print p2.name  # aaa
print Person.name  # aaa
```

**类变量就是供类使用的变量，实例变量就是供实例使用的**。

这里 `p1.name="bbb"` 是实例调用了类变量，这其实和上面第一个问题一样，就是函数传参的问题， `p1.name` 一开始是指向的类变量 `name="aaa"` ，但是在实例的作用域里把类变量的引用改变了，就变成了一个实例变量，`self.name` 不再引用 Person 的类变量 name 了.

可以看看下面的例子:

```python
class Person:
    name=[]
 
p1=Person()
p2=Person()
p1.name.append(1)
print p1.name  # [1]
print p2.name  # [1]
print Person.name  # [1]
```

[参考](http://stackoverflow.com/questions/6470428/catch-multiple-exceptions-in-one-line-except-block)

### 1.5 Python自省

这个也是 python 彪悍的特性.

**自省就是面向对象的语言所写的程序在运行时，所能知道对象的类型。简单一句就是运行时能够获得对象的类型。比如 type(), dir(), getattr(), hasattr(), isinstance()**.

### 1.6 字典推导式

可能你见过列表推导时，却没有见过字典推导式，在2.7中才加入的：

```python
d = {key: value for (key, value) in iterable}
```

### 1.7 Python中单下划线和双下划线

```python
>>> class MyClass():
...     def __init__(self):
...             self.__superprivate = "Hello"
...             self._semiprivate = ", world!"
...
>>> mc = MyClass()
>>> print mc.__superprivate
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: myClass instance has no attribute '__superprivate'
>>> print mc._semiprivate
, world!
>>> print mc.__dict__
{'_MyClass__superprivate': 'Hello', '_semiprivate': ', world!'}
```

- `__foo__`：一种约定，Python 内部的名字，用来区别其他用户自定义的命名，以防冲突.

- `_foo`：一种约定，用来指定变量私有。程序员用来指定私有变量的一种方式.

- `__foo`：这个有真正的意义：解析器用 `_classname__foo` 来代替这个名字，以区别和其他类相同的命名.

[详情见](http://stackoverflow.com/questions/1301346/the-meaning-of-a-single-and-a-double-underscore-before-an-object-name-in-python)

[或者](http://www.zhihu.com/question/19754941)

### 1.8 字符串格式化: % 和 .format

`.format` 在许多方面看起来更便利。对于 `%` 最烦人的是它无法同时传递一个变量和元组。你可能会想下面的代码不会有什么问题：

```python
"hi there %s" % name
```

但是，如果 name 恰好是 (1,2,3)，它将会抛出一个 TypeError 异常。为了保证它总是正确的，你必须这样做：

```python
"hi there %s" % (name,)   # 提供一个单元素的数组而不是一个参数
```

但是有点丑。 `.format` 就没有这些问题。你给的第二个问题也是这样，`.format` 好看多了.

你为什么不用它?

- 不知道它(在读这个之前)
- 为了和 Python2.5 兼容(譬如logging库建议使用`%`([issue #4](https://github.com/taizilongxu/interview_python/issues/4)))

http://stackoverflow.com/questions/5082452/python-string-formatting-vs-format

### 1.9 迭代器和生成器

[这个是 stackoverflow 里 python 排名第一的问题，值得一看:](http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python)

[这是中文版](http://taizilongxu.gitbooks.io/stackoverflow-about-python/content/1/README.html)

### 1.10 `*args` and `**kwargs`

用 `*args` 和 `**kwargs` 只是为了方便并没有强制使用它们.

**当你不确定你的函数里将要传递多少参数时你可以用 `*args`**。例如，它可以传递任意数量的参数：

```python
>>> def print_everything(*args):
        for count, thing in enumerate(args):
...         print '{0}. {1}'.format(count, thing)
...
>>> print_everything('apple', 'banana', 'cabbage')
0. apple
1. banana
2. cabbage
```

相似的，`**kwargs` 允许你使用没有事先定义的参数名：

```python
>>> def table_things(**kwargs):
...     for name, value in kwargs.items():
...         print '{0} = {1}'.format(name, value)
...
>>> table_things(apple = 'fruit', cabbage = 'vegetable')
cabbage = vegetable
apple = fruit
```

你也可以混着用。命名参数首先获得参数值然后所有的其他参数都传递给 `*args` 和 `**kwargs`。命名参数在列表的最前端。例如：

```python
def table_things(titlestring, **kwargs)
```

`*args` 和 `**kwargs` 可以同时在函数的定义中，但是 `*args` 必须在 `**kwargs` 前面.

当调用函数时你也可以用 `*` 和 `**` 语法。例如：

```python
>>> def print_three_things(a, b, c):
...     print 'a = {0}, b = {1}, c = {2}'.format(a,b,c)
...
>>> mylist = ['aardvark', 'baboon', 'cat']
>>> print_three_things(*mylist)
 
a = aardvark, b = baboon, c = cat
```

就像你看到的一样，它可以传递列表 (或者元组) 的每一项并把它们解包.注意必须与它们在函数里的参数相吻合。当然，你也可以在函数定义或者函数调用时用`*`。

http://stackoverflow.com/questions/3394835/args-and-kwargs

### 1.11 面向切面编程 AOP 和装饰器

这个 AOP 一听起来有点懵，同学面阿里的时候就被问懵了…

**装饰器是一个很著名的设计模式，经常被用于有切面需求的场景，较为经典的有插入日志、性能测试、事务处理等**。

装饰器是解决这类问题的绝佳设计，有了装饰器，我们就可以抽离出大量函数中与函数功能本身无关的雷同代码并继续重用。概括的讲，**装饰器的作用就是为已经存在的对象添加额外的功能。**

[这个问题比较大,推荐:](http://stackoverflow.com/questions/739654/how-can-i-make-a-chain-of-function-decorators-in-python)

[中文](http://taizilongxu.gitbooks.io/stackoverflow-about-python/content/3/README.html)

### 1.12 鸭子类型

> 当看到一只鸟走起来像鸭子、游泳起来像鸭子、叫起来也像鸭子，那么这只鸟就可以被称为鸭子。

我们并不关心对象是什么类型，到底是不是鸭子，只关心行为。

比如在 python 中，有很多 file-like 的东西，比如 StringIO，GzipFile，socket。它们有很多相同的方法，我们把它们当作文件使用。

又比如 `list.extend()` 方法中，我们并不关心它的参数是不是 list，只要它是可迭代的，所以它的参数可以是 `list/tuple/dict/字符串/生成器等`.

鸭子类型在动态语言中经常使用，非常灵活，使得 python 不像 java 那样专门去弄一大堆的设计模式。

### 1.13 Python中重载

[引自知乎](http://www.zhihu.com/question/20053359)

**函数重载主要是为了解决两个问题**。

1. 可变参数类型。
2. 可变参数个数。

另外，一个基本的设计原则是，仅仅当两个函数除了参数类型和参数个数不同以外，其功能是完全相同的，此时才使用函数重载，**如果两个函数的功能其实不同，那么不应当使用重载，而应当使用一个名字不同的函数**。

好吧，那么对于情况 1 ，函数功能相同，但是参数类型不同，python 如何处理？答案是根本不需要处理，**因为 python 可以接受任何类型的参数，如果函数的功能相同，那么不同的参数类型在 python 中很可能是相同的代码，没有必要做成两个不同函数**。

那么对于情况 2 ，函数功能相同，但参数个数不同，python 如何处理？大家知道，答案就是缺省参数。对那些缺少的参数设定为缺省参数即可解决问题。因为你假设函数功能相同，那么那些缺少的参数终归是需要用的。

好了，鉴于情况 1 跟 情况 2 都有了解决方案，**python 自然就不需要函数重载了**。

### 1.14 新式类和旧式类

这个面试官问了，我说了老半天，不知道他问的真正意图是什么.

[stackoverflow](http://stackoverflow.com/questions/54867/what-is-the-difference-between-old-style-and-new-style-classes-in-python)

这篇文章很好的介绍了新式类的特性: http://www.cnblogs.com/btchenguang/archive/2012/09/17/2689146.html

新式类很早在 2.2 就出现了，所以旧式类完全是兼容的问题，**Python3 里的类全部都是新式类**。这里有一个 MRO 问题可以了解下 (新式类是广度优先,旧式类是深度优先) ，<Python核心编程>里讲的也很多.

### 1.15 `__new__`和`__init__`的区别

这个 `__new__` 确实很少见到，先做了解吧.

1. `__new__` 是一个静态方法，而 `__init__` 是一个实例方法.
2. `__new__` 方法会返回一个创建的实例，而 `__init__` 什么都不返回.
3. 只有在 `__new__` 返回一个 cls 的实例时后面的 `__init__` 才能被调用.
4. 当创建一个新实例时调用 `__new__` ，初始化一个实例时用 `__init__`.

[stackoverflow](http://stackoverflow.com/questions/674304/pythons-use-of-new-and-init)

ps： `__metaclass__` 是创建类时起作用。所以我们可以分别使用 `__metaclass__`，`__new__` 和 `__init__` 来分别在类创建，实例创建和实例初始化的时候做一些小手脚.

### 1.16 单例模式

这个绝对常考啊。绝对要记住 1~2 个方法，当时面试官是让手写的。

#### 1.16.1 使用`__new__`方法

```python
class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance
 
class MyClass(Singleton):
    a = 1
```

#### 1.16.2 共享属性

创建实例时把所有实例的 `__dict__` 指向同一个字典，这样它们具有相同的属性和方法.

```python
class Borg(object):
    _state = {}
    def __new__(cls, *args, **kw):
        ob = super(Borg, cls).__new__(cls, *args, **kw)
        ob.__dict__ = cls._state
        return ob
 
class MyClass2(Borg):
    a = 1
```

#### 1.16.3 装饰器版本

```python
def singleton(cls, *args, **kw):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return getinstance
 
@singleton
class MyClass:
  ...
```

#### 1.16.4 import方法

作为 python 的模块是天然的单例模式

```python
# mysingleton.py
class My_Singleton(object):
    def foo(self):
        pass
 
my_singleton = My_Singleton()
 
# to use
from mysingleton import my_singleton
 
my_singleton.foo()
```

### 1.17 Python 中的作用域

Python 中，一个变量的作用域总是由在代码中被赋值的地方所决定的。

当 Python 遇到一个变量的话他会按照这样的顺序进行搜索：

本地作用域（Local）→当前作用域被嵌入的本地作用域（Enclosing locals）→全局/模块作用域（Global）→内置作用域（Built-in）

### 1.18 GIL 线程全局锁

线程全局锁(Global Interpreter Lock)，即 Python 为了保证线程安全而采取的独立线程运行的限制，说白了就是一个核只能在同一时间运行一个线程.

见[Python 最难的问题](http://www.oschina.net/translate/pythons-hardest-problem)

**解决办法就是多进程和下面的协程(协程也只是单 CPU，但是能减小切换代价提升性能)**.

### 1.19 协程

知乎被问到了，呵呵哒，跪了

**简单点说协程是进程和线程的升级版，进程和线程都面临着内核态和用户态的切换问题而耗费许多切换时间，而协程就是用户自己控制切换的时机，不再需要陷入系统的内核态**.

**Python 里最常见的 yield 就是协程的思想**! 可以查看第九个问题.

### 1.20 闭包

闭包(closure)是函数式编程的重要的语法结构。闭包也是一种组织代码的结构，它同样提高了代码的可重复使用性。

当一个内嵌函数引用其外部作作用域的变量，我们就会得到一个闭包.。总结一下，**创建一个闭包必须满足以下几点**：

1. **必须有一个内嵌函数**
2. **内嵌函数必须引用外部函数中的变量**
3. **外部函数的返回值必须是内嵌函数**

感觉闭包还是有难度的，几句话是说不明白的，还是查查相关资料.

重点是函数运行后并不会被撤销，就像 16 题的 instance 字典一样，当函数运行完后，instance 并不被销毁，而是继续留在内存空间里。这个功能类似类里的类变量，只不过迁移到了函数上.

闭包就像个空心球一样，你知道外面和里面，但你不知道中间是什么样.

### 1.21 lambda函数

其实就是一个匿名函数,为什么叫 lambda ? 因为和后面的函数式编程有关.

推荐: [知乎](http://www.zhihu.com/question/20125256)

### 1.22 Python函数式编程

这个需要适当的了解一下吧，毕竟函数式编程在 Python 中也做了引用.

推荐: [酷壳](http://coolshell.cn/articles/10822.html)

python 中函数式编程支持:

`filter` 函数的功能相当于过滤器。调用一个布尔函数 `bool_func` 来迭代遍历每个 seq 中的元素；返回一个使 `bool_seq` 返回值为 true 的元素的序列。

```python
>>>a = [1,2,3,4,5,6,7]
>>>b = filter(lambda x: x > 5, a)
>>>print b
>>>[6,7]
```

map 函数是对一个序列的每个项依次执行函数，下面是对一个序列每个项都乘以 2：

```python
>>> a = map(lambda x:x*2,[1,2,3])
>>> list(a)
[2, 4, 6]
```

reduce 函数是对一个序列的每个项迭代调用函数，下面是求 3 的阶乘：

```python
>>> reduce(lambda x,y:x*y,range(1,4))
6
```

### 1.23 Python里的拷贝

引用 和 copy()，deepcopy() 的区别

```python
import copy
a = [1, 2, 3, 4, ['a', 'b']]  #原始对象
 
b = a  #赋值，传对象的引用
c = copy.copy(a)  #对象拷贝，浅拷贝
d = copy.deepcopy(a)  #对象拷贝，深拷贝
 
a.append(5)  #修改对象a
a[4].append('c')  #修改对象a中的['a', 'b']数组对象
 
print 'a = ', a
print 'b = ', b
print 'c = ', c
print 'd = ', d
 
输出结果：
a =  [1, 2, 3, 4, ['a', 'b', 'c'], 5]
b =  [1, 2, 3, 4, ['a', 'b', 'c'], 5]
c =  [1, 2, 3, 4, ['a', 'b', 'c']]
d =  [1, 2, 3, 4, ['a', 'b']]
```

### 1.24 Python垃圾回收机制

**Python GC 主要使用引用计数（reference counting）来跟踪和回收垃圾**。**在引用计数的基础上，通过 “标记-清除”（mark and sweep）解决容器对象可能产生的循环引用问题，通过 “分代回收”（generation collection）以空间换时间的方法提高垃圾回收效率**。

#### 1.24.1 引用计数

PyObject 是每个对象必有的内容，其中 `ob_refcnt` 就是做为引用计数。当一个对象有新的引用时，它的 `ob_refcnt` 就会增加，当引用它的对象被删除，它的 `ob_refcnt` 就会减少.引用计数为 0 时，该对象生命就结束了。

**优点**:

1. 简单
2. 实时性

**缺点**:

1. 维护引用计数消耗资源
2. 循环引用

#### 1.24.2 标记-清除机制

基本思路是先按需分配，等到没有空闲内存的时候从寄存器和程序栈上的引用出发，遍历以对象为节点、以引用为边构成的图，把所有可以访问到的对象打上标记，然后清扫一遍内存空间，把所有没标记的对象释放。

#### 1.24.3 分代技术

分代回收的整体思想是：将系统中的所有内存块根据其存活时间划分为不同的集合，每个集合就成为一个 “代”，垃圾收集频率随着 “代” 的存活时间的增大而减小，存活时间通常利用经过几次垃圾回收来度量。

**Python默认定义了三代对象集合，索引数越大，对象存活时间越长**。

举例：

> 当某些内存块 M 经过了 3 次垃圾收集的清洗之后还存活时，我们就将内存块 M 划到一个集合 A 中去，而新分配的内存都划分到集合 B 中去。当垃圾收集开始工作时，大多数情况都只对集合 B 进行垃圾回收，而对集合 A 进行垃圾回收要隔相当长一段时间后才进行，这就使得垃圾收集机制需要处理的内存少了，效率自然就提高了。在这个过程中，集合 B 中的某些内存块由于存活时间长而会被转移到集合A中，当然，集合 A 中实际上也存在一些垃圾，这些垃圾的回收会因为这种分代的机制而被延迟。

### 1.25 Python的List

推荐: http://www.jianshu.com/p/J4U6rR

### 1.26 Python的is

**`is` 是对比地址，`==` 是对比值**

### 1.27 read, readline 和 readlines

- read 读取整个文件
- readline 读取下一行,使用生成器方法
- readlines 读取整个文件到一个迭代器以供我们遍历

### 1.28 Python2 和 3 的区别

推荐：《[Python 2.7.x 和 3.x 版本的重要区别](http://python.jobbole.com/80006/)》

## 2. 操作系统

### 2.1 select, poll 和 epoll

**其实所有的 I/O 都是轮询的方法，只不过实现的层面不同罢了**.

这个问题可能有点深入了，但相信能回答出这个问题是对 I/O 多路复用有很好的了解了。其中 tornado 使用的就是 epoll 的.

[selec,poll和epoll区别总结](http://www.cnblogs.com/Anker/p/3265058.html)

基本上 select 有 3 个缺点：

1. 连接数受限
2. 查找配对速度慢
3. 数据由内核拷贝到用户态

poll 改善了第一个缺点

epoll 改了三个缺点.

关于epoll的: http://www.cnblogs.com/my_life/articles/3968782.html

### 2.2 调度算法

1. 先来先服务(FCFS, First Come First Serve)
2. 短作业优先(SJF, Shortest Job First)
3. 最高优先权调度(Priority Scheduling)
4. 时间片轮转(RR, Round Robin)
5. 多级反馈队列调度(multilevel feedback queue scheduling)

**实时调度算法**：

1. 最早截至时间优先 EDF
2. 最低松弛度优先 LLF

### 2.3 死锁

**原因**：

1. 竞争资源
2. 程序推进顺序不当

**必要条件**：

1. 互斥条件
2. 请求和保持条件
3. 不剥夺条件
4. 环路等待条件

**处理死锁基本方法**：

1. 预防死锁 (摒弃除 1 以外的条件)
2. 避免死锁 (银行家算法)
3. 检测死锁 (资源分配图)
4. 解除死锁
   1. 剥夺资源
   2. 撤销进程

### 2.4 程序编译与链接

推荐: http://www.ruanyifeng.com/blog/2014/11/compiler.html

Bulid 过程可以分解为 4 个步骤：

- 预处理(Prepressing),
- 编译(Compilation)
- 汇编(Assembly)
- 链接(Linking)

以 c 语言为例：

#### 2.4.1 预处理

预编译过程主要处理那些源文件中的以 “#” 开始的预编译指令，主要处理规则有：

1. 将所有的 “#define” 删除，并展开所用的宏定义
2. 处理所有条件预编译指令，比如 “#if”、“#ifdef”、 “#elif”、“#endif”
3. 处理 “#include” 预编译指令，将被包含的文件插入到该编译指令的位置，注：此过程是递归进行的
4. 删除所有注释
5. 添加行号和文件名标识，以便于编译时编译器产生调试用的行号信息以及用于编译时产生编译错误或警告时可显示行号
6. 保留所有的 #pragma 编译器指令。

#### 2.4.2 编译

编译过程就是把预处理完的文件进行一系列的词法分析、语法分析、语义分析及优化后生成相应的汇编代码文件。这个过程是整个程序构建的核心部分。

#### 2.4.3 汇编

汇编器是将汇编代码转化成机器可以执行的指令，每一条汇编语句几乎都是一条机器指令。经过编译、链接、汇编输出的文件成为目标文件 (Object File)

#### 2.4.4 链接

链接的主要内容就是把各个模块之间相互引用的部分处理好，使各个模块可以正确的拼接。

链接的主要过程包块地址和空间的分配（Address and Storage Allocation）、符号决议(Symbol Resolution)和重定位 (Relocation)等步骤。

### 2.5 静态链接和动态链接

**静态链接方法**：

- 静态链接的时候，载入代码就会把程序会用到的动态代码或动态代码的地址确定下来
- 静态库的链接可以使用静态链接，动态链接库也可以使用这种方法链接导入库

**动态链接方法**：

- 使用这种方式的程序并不在一开始就完成动态链接，而是直到真正调用动态库代码时，载入程序才计算 (被调用的那部分) 动态代码的逻辑地址，然后等到某个时候，程序又需要调用另外某块动态代码时，载入程序又去计算这部分代码的逻辑地址，所以，这种方式使程序初始化时间较短，但运行期间的性能比不上静态链接的程序

### 2.6 虚拟内存技术

虚拟存储器是值具有请求调入功能和置换功能，能从逻辑上对内存容量加以扩充的一种存储系统.

### 2.7 分页和分段

**分页**：

- 用户程序的地址空间被划分成若干固定大小的区域，称为 “页”，相应地，内存空间分成若干个物理块，页和块的大小相等。可将用户程序的任一页放在内存的任一块中，实现了离散分配。

**分段**：

- 将用户程序地址空间分成若干个大小不等的段，每段可以定义一组相对完整的逻辑信息。存储分配时，以段为单位，段与段在内存中可以不相邻接，也实现了离散分配。

#### 2.7.1 分页与分段的主要区别

1. 页是信息的物理单位，分页是为了实现非连续分配，以便解决内存碎片问题，或者说分页是由于系统管理的需要。段是信息的逻辑单位，它含有一组意义相对完整的信息，分段的目的是为了更好地实现共享，满足用户的需要.
2. 页的大小固定，由系统确定，将逻辑地址划分为页号和页内地址是由机器硬件实现的。而段的长度却不固定，决定于用户所编写的程序，通常由编译程序在对源程序进行编译时根据信息的性质来划分.
3. 分页的作业地址空间是一维的。分段的地址空间是二维的.

### 2.8 页面置换算法

1. 最佳置换算法 OPT：不可能实现
2. 先进先出 FIFO
3. 最近最久未使用算法 LRU：最近一段时间里最久没有使用过的页面予以置换.
4. clock 算法

### 2.9 边沿触发和水平触发

- 边缘触发是指每当状态变化时发生一个 io 事件
- 条件触发是只要满足条件就发生一个 io 事件

## 3. 数据库

### 3.1 事务

数据库事务(Database Transaction) ，是指作为单个逻辑工作单元执行的一系列操作，要么完全地执行，要么完全地不执行。

### 3.2 数据库索引

推荐: http://tech.meituan.com/mysql-index.html

[MySQL索引背后的数据结构及算法原理](http://blog.jobbole.com/24006/)

聚集索引，非聚集索引，B-Tree，B+Tree，最左前缀原理

### 3.3 Redis原理

TODO

### 3.4 乐观锁和悲观锁

**悲观锁**：

- 假定会发生并发冲突，屏蔽一切可能违反数据完整性的操作

**乐观锁**：

- 假设不会发生并发冲突，只在提交操作时检查是否违反数据完整性。

### 3.5 MVCC

TODO

### 3.6 MyISAM 和 InnoDB

**MyISAM**： 适合于一些需要大量查询的应用，但其对于有大量写操作并不是很好。甚至你只是需要 update 一个字段，整个表都会被锁起来，而别的进程，就算是读进程都无法操作直到读操作完成。另外，MyISAM 对于 SELECT COUNT(*) 这类的计算是超快无比的。

**InnoDB**： 的趋势会是一个非常复杂的存储引擎，对于一些小的应用，它会比 MyISAM 还慢。他是它支持 “行锁” ，于是在写操作比较多的时候，会更优秀。并且，他还支持更多的高级应用，比如：事务。

## 4. 网络

### 4.1 三次握手

1. 客户端通过向服务器端发送一个 SYN 来创建一个主动打开，作为三路握手的一部分。客户端把这段连接的序号设定为随机数 A。
2. 服务器端应当为一个合法的 SYN 回送一个 SYN/ACK。ACK 的确认码应为 A+1，SYN/ACK 包本身又有一个随机序号 B。
3. 最后，客户端再发送一个 ACK。当服务端受到这个 ACK 的时候，就完成了三路握手，并进入了连接创建状态。此时包序号被设定为收到的确认号 A+1，而响应则为 B+1。

### 4.2 四次挥手

TODO

### 4.3 ARP协议

**地址解析协议(Address Resolution Protocol)**: 

- 根据 IP 地址获取物理地址的一个 TCP/IP 协议

### 4.4 urllib 和 urllib2 的区别

这个面试官确实问过，当时答的 urllib2 可以 Post 而 urllib 不可以.

1. urllib 提供 urlencode 方法用来 GET 查询字符串的产生，而 urllib2 没有。这是为何 urllib 常和 urllib2 一起使用的原因。
2. urllib2 可以接受一个 Reques t类的实例来设置 URL 请求的 headers，urllib 仅可以接受 URL。这意味着，你不可以伪装你的 User Agent 字符串等。

### 4.5 Post 和 Get

[GET和POST有什么区别？及为什么网上的多数答案都是错的](http://www.cnblogs.com/nankezhishi/archive/2012/06/09/getandpost.html)

get: [RFC 2616 – Hypertext Transfer Protocol — HTTP/1.1](http://tools.ietf.org/html/rfc2616#section-9.3)
post: [RFC 2616 – Hypertext Transfer Protocol — HTTP/1.1](http://tools.ietf.org/html/rfc2616#section-9.5)

### 4.6 Cookie 和 Session

|          | Cookie                                               | Session  |
| -------- | ---------------------------------------------------- | -------- |
| 储存位置 | 客户端                                               | 服务器端 |
| 目的     | 跟踪会话，也可以保存用户偏好设置或者保存用户名密码等 | 跟踪会话 |
| 安全性   | 不安全                                               | 安全     |

**session 技术是要使用到 cookie 的，之所以出现 session 技术，主要是为了安全**。

### 4.7 apache和nginx的区别

**nginx 相对 apache 的优点**：

- 轻量级，同样起web 服务，比apache 占用更少的内存及资源
- 抗并发，nginx 处理请求是异步非阻塞的，支持更多的并发连接，而apache 则是阻塞型的，在高并发下nginx 能保持低资源低消耗高性能
- 配置简洁
- 高度模块化的设计，编写模块相对简单
- 社区活跃

**apache 相对nginx 的优点**：

- rewrite ，比nginx 的rewrite 强大
- 模块超多，基本想到的都可以找到
- 少bug ，nginx 的bug 相对较多
- 超稳定

### 4.8 网站用户密码保存

1. 明文保存
2. 明文 hash 后保存，如 md5
3. MD5 + Salt 方式,这个 salt 可以随机
4. 知乎使用了 Bcrypy(好像)加密

### 4.9 HTTP和HTTPS

| 状态码         | 定义                            |
| -------------- | ------------------------------- |
| 1xx 报告       | 接收到请求，继续进程            |
| 2xx 成功       | 步骤成功接收，被理解，并被接受  |
| 3xx 重定向     | 为了完成请求,必须采取进一步措施 |
| 4xx 客户端出错 | 请求包括错的顺序或不能完成      |
| 5xx 服务器出错 | 服务器无法完成显然有效的请求    |

403: Forbidden
404: Not Found

HTTPS 握手，对称加密，非对称加密，TLS/SSL，RSA

### 4.10 XSRF 和 XSS

- CSRF(Cross-site request forgery)跨站请求伪造
- XSS(Cross Site Scripting)跨站脚本攻击

**CSRF 重点在请求，XSS重点在脚本**

### 4.11 幂等 Idempotence

HTTP 方法的幂等性是指一次和多次请求某一个资源应该具有同样的**副作用**。(注意是副作用)

`GET http://www.bank.com/account/123456`，不会改变资源的状态，不论调用一次还是 N 次都没有副作用。请注意，这里强调的是一次和 N 次具有相同的副作用，而不是每次 GET 的结果相同。`GET http://www.news.com/latest-news`这个 HTTP 请求可能会每次得到不同的结果，但它本身并没有产生任何副作用，因而是满足幂等性的。

**DELETE**： 方法用于删除资源，有副作用，但它应该满足幂等性。比如：`DELETE http://www.forum.com/article/4231`，调用一次和 N 次对系统产生的副作用是相同的，即删掉 id 为 4231 的帖子；因此，调用者可以多次调用或刷新页面而不必担心引起错误。

**POST**： 所对应的 URI 并非创建的资源本身，而是资源的接收者。比如：`POST http://www.forum.com/articles`的语义是在`http://www.forum.com/articles`下创建一篇帖子，HTTP响应中应包含帖子的创建状态以及帖子的 URI。两次相同的 POST 请求会在服务器端创建两份资源，它们具有不同的 URI；所以，POST 方法不具备幂等性。

**PUT**： 所对应的 URI 是要创建或更新的资源本身。比如：`PUT http://www.forum/articles/4231`的语义是创建或更新 ID 为 4231 的帖子。对同一 URI 进行多次 PUT 的副作用和一次 PUT 是相同的；因此，PUT 方法具有幂等性。

### 4.12 RESTful架构(SOAP,RPC)

推荐: http://www.ruanyifeng.com/blog/2011/09/restful.html

### 4.13 SOAP

SOAP（原为Simple Object Access Protocol的首字母缩写，即简单对象访问协议）是交换数据的一种协议规范，使用在计算机网络 Web 服务（web service）中，交换带结构信息。SOAP 为了简化网页服务器（Web Server）从 XML 数据库中提取数据时，节省去格式化页面时间，以及不同应用程序之间按照 HTTP 通信协议，遵从 XML 格式执行资料互换，使其抽象于语言实现、平台和硬件。

### 4.14 RPC

RPC（Remote Procedure Call Protocol）——远程过程调用协议，它是一种通过网络从远程计算机程序上请求服务，而不需要了解底层网络技术的协议。RPC 协议假定某些传输协议的存在，如 TCP 或 UDP，为通信程序之间携带信息数据。在 OSI 网络通信模型中，RPC 跨越了传输层和应用层。RPC 使得开发包括网络分布式多程序在内的应用程序更加容易。

**总结**：服务提供的两大流派。传统意义以方法调用为导向通称 RPC。为了企业 SOA，若干厂商联合推出webservice，制定了 wsdl 接口定义，传输 soap。当互联网时代，臃肿 SOA 被简化为 http+xml/json。但是简化出现各种混乱。以资源为导向，任何操作无非是对资源的增删改查，于是统一的 REST 出现了.

**进化的顺序: RPC -> SOAP -> RESTful**

### 4.15 CGI 和 WSGI

- CGI 是通用网关接口，是连接 web 服务器和应用程序的接口，用户通过 CGI 来获取动态数据或文件等。

  CGI 程序是一个独立的程序，它可以用几乎所有语言来写，包括 perl，c，lua，python 等等。

- WSGI, Web Server Gateway Interface，是 Python 应用程序或框架和 Web 服务器之间的一种接口，WSGI 的其中一个目的就是让用户可以用统一的语言(Python)编写前后端。

官方说明：[PEP-3333](https://www.python.org/dev/peps/pep-3333/)

### 4.16 中间人攻击

在 GFW 里屡见不鲜的，呵呵.

中间人攻击（Man-in-the-middle attack，通常缩写为 MITM）是指攻击者与通讯的两端分别创建独立的联系，并交换其所收到的数据，使通讯的两端认为他们正在通过一个私密的连接与对方直接对话，但事实上整个会话都被攻击者完全控制。

### 4.17 c10k 问题

所谓 c10k 问题，指的是服务器同时支持成千上万个客户端的问题，也就是 concurrent 10 000 connection（这也是 c10k 这个名字的由来）。
推荐: http://www.kegel.com/c10k.html

### 4.18 socket

推荐: http://www.cnblogs.com/bingyun84/archive/2009/10/16/1584387.html

**Socket = Ip address + TCP/UDP + port**

### 4.19 浏览器缓存

推荐: <http://web.jobbole.com/84367/>

304 Not Modified

### 4.20 HTTP1.0和HTTP1.1

推荐: http://blog.csdn.net/elifefly/article/details/3964766

1. 请求头 Host 字段，一个服务器多个网站
2. 长链接
3. 文件断点续传
4. 身份认证，状态管理，Cache缓存

### 4.21 Ajax

AJAX，Asynchronous JavaScript and XML（异步的 JavaScript 和 XML）, 是与在不重新加载整个页面的情况下，与服务器交换数据并更新部分网页的技术。

## 5. *NIX

### 5.1 unix进程间通信方式(IPC)

1. 管道（Pipe）：管道可用于具有亲缘关系进程间的通信，允许一个进程和另一个与它有共同祖先的进程之间进行通信。
2. 命名管道（named pipe）：命名管道克服了管道没有名字的限制，因此，除具有管道所具有的功能外，它还允许无亲缘关系进程间的通信。命名管道在文件系统中有对应的文件名。命名管道通过命令 mkfifo 或系统调用 mkfifo 来创建。
3. 信号（Signal）：信号是比较复杂的通信方式，用于通知接受进程有某种事件发生，除了用于进程间通信外，进程还可以发送信号给进程本身；linux 除了支持 Unix 早期信号语义函数 sigal 外，还支持语义符合 Posix.1 标准的信号函数 sigaction（实际上，该函数是基于 BSD 的，BSD 为了实现可靠信号机制，又能够统一对外接口，用 sigaction 函数重新实现了 signal 函数）。
4. 消息（Message）队列：消息队列是消息的链接表，包括 Posix 消息队列 system V 消息队列。有足够权限的进程可以向队列中添加消息，被赋予读权限的进程则可以读走队列中的消息。消息队列克服了信号承载信息量少，管道只能承载无格式字节流以及缓冲区大小受限等缺
5. 共享内存：使得多个进程可以访问同一块内存空间，是最快的可用 IPC 形式。是针对其他通信机制运行效率较低而设计的。往往与其它通信机制，如信号量结合使用，来达到进程间的同步及互斥。
6. 内存映射（mapped memory）：内存映射允许任何多个进程间通信，每一个使用该机制的进程通过把一个共享的文件映射到自己的进程地址空间来实现它。
7. 信号量（semaphore）：主要作为进程间以及同一进程不同线程之间的同步手段。
8. 套接口（Socket）：更为一般的进程间通信机制，可用于不同机器之间的进程间通信。起初是由 Unix 系统的 BSD 分支开发出来的，但现在一般可以移植到其它类 Unix 系统上：Linux 和 System V 的变种都支持套接字。

## 6. 数据结构

### 6.1 红黑树

红黑树与 AVL 的比较：

- AVL 是严格平衡树，因此在增加或者删除节点的时候，根据不同情况，旋转的次数比红黑树要多；

- 红黑是用非严格的平衡来换取增删节点时候旋转次数的降低；

所以简单说：

- 如果你的应用中，搜索的次数远远大于插入和删除，那么选择 AVL，
- 如果搜索，插入删除次数几乎差不多，应该选择 RB。

## 7. 编程题

### 7.1 台阶问题/斐波纳挈

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

```python
fib = lambda n: n if n <= 2 else fib(n - 1) + fib(n - 2)
```

第二种记忆方法

```python
def memo(func):
    cache = {}
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrap
 
 
@ memo
def fib(i):
    if i < 2:
        return 1
    return fib(i-1) + fib(i-2)
```

第三种方法

```python
def fib(n):
    a, b = 0, 1
    for _ in xrange(n):
        a, b = b, a + b
    return b
```

### 7.2 变态台阶问题

一只青蛙一次可以跳上 1 级台阶，也可以跳上 2 级……它也可以跳上 n 级。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

```python
fib = lambda n: n if n < 2 else 2 * fib(n - 1)
```

### 7.3 矩形覆盖

我们可以用 `2*1` 的小矩形横着或者竖着去覆盖更大的矩形。请问用 n 个 `2*1` 的小矩形无重叠地覆盖一个 `2*n` 的大矩形，总共有多少种方法？

> 第 `2*n` 个矩形的覆盖方法等于第 `2*(n-1)` 加上第 `2*(n-2)` 的方法。

```python
f = lambda n: 1 if n < 2 else f(n - 1) + f(n - 2)
```

### 7.4 杨氏矩阵查找

在一个 m 行 n 列二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

```python
# TODO
```

### 7.5 去除列表中的重复元素

用集合

```python
list(set(L))
```

用字典

```python
l1 = ['b','c','d','b','c','a','a']
l2 = {}.fromkeys(l1).keys()
print l2
```

用字典并保持顺序

```python
l1 = ['b','c','d','b','c','a','a']
l2 = list(set(l1))
l2.sort(key=l1.index)
print l2
```

列表推导式

```python
l1 = ['b','c','d','b','c','a','a']
l2 = []
[l2.append(i) for i in l1 if not i in l2]
```

面试官提到的,先排序然后删除.

### 7.6 链表成对调换

`1->2->3->4` 转换成 `2->1->4->3`.

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
 
class Solution:
    # @param a ListNode
    # @return a ListNode
    def swapPairs(self, head):
        if head != None and head.next != None:
            next = head.next
            head.next = self.swapPairs(next.next)
            next.next = head
            return next
        return head
```

### 7.7 创建字典的方法

#### 7.7.1 直接创建

```python
dict = {'name':'earth', 'port':'80'}
```

#### 7.7.2 工厂方法

```python
items=[('name','earth'),('port','80')]
dict2=dict(items)
dict1=dict((['name','earth'],['port','80']))
```

#### 7.7.3 fromkeys()方法

```python
dict1={}.fromkeys(('x','y'),-1)
dict={'x':-1,'y':-1}
dict2={}.fromkeys(('x','y'))
dict2={'x':None, 'y':None}
```

### 7.8 合并两个有序列表

知乎远程面试要求编程

尾递归

```python
def _recursion_merge_sort2(l1, l2, tmp):
    if len(l1) == 0 or len(l2) == 0:
        tmp.extend(l1)
        tmp.extend(l2)
        return tmp
    else:
        if l1[0] < l2[0]:
            tmp.append(l1[0])
            del l1[0]
        else:
            tmp.append(l2[0])
            del l2[0]
        return _recursion_merge_sort2(l1, l2, tmp)
 
def recursion_merge_sort2(l1, l2):
    return _recursion_merge_sort2(l1, l2, [])
```

循环算法

```python
def loop_merge_sort(l1, l2):
    tmp = []
    while len(l1) > 0 and len(l2) > 0:
        if l1[0] < l2[0]:
            tmp.append(l1[0])
            del l1[0]
        else:
            tmp.append(l2[0])
            del l2[0]
    tmp.extend(l1)
    tmp.extend(l2)
    return tmp
```

### 7.9 交叉链表求交点

去哪儿的面试，没做出来.

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def node(l1, l2):
    length1, lenth2 = 0, 0
    # 求两个链表长度
    while l1.next:
        l1 = l1.next
        length1 += 1
    while l2.next:
        l2 = l2.next
        length2 += 1
    # 长的链表先走
    if length1 > lenth2:
        for _ in range(length1 - length2):
            l1 = l1.next
    else:
        for _ in range(length2 - length1):
            l2 = l2.next
    while l1 and l2:
        if l1.next == l2.next:
            return l1.next
        else:
            l1 = l1.next
            l2 = l2.next
```

### 7.10 二分查找

```python
def binary_search(alist, item):
    """
    :type item: int
    :type alist: list
    :rtype bool
    """
	n = len(alist)
    start = 0
    end = n - 1
    while start <= end:
        mid = (start + end) // 2
        if alist[mid] == item:
            return True
        elif alist[mid] > item:
            end = mid -1
        else:
            start = mid + 1
    
    return False

# 递归方式实现
def binary_search(alist, item):
    n = len(alist)
    if 0 == n:
        return False
    mid = n // 2
    if alist[mid] == item:
        return True
    elif alist[mid] > item:
        return binary_search(alist[:mid], item)
    else:
        return binary_search(alist[mid+1:], item) 
```

### 7.11 快排

```python
def qsort(seq):
    if seq==[]:
        return []
    else:
        pivot=seq[0]
        lesser=qsort([x for x in seq[1:] if x<pivot])
        greater=qsort([x for x in seq[1:] if x>=pivot])
        return lesser+[pivot]+greater
 
if __name__=='__main__':
    seq=[5,6,78,9,0,-1,2,3,-65,12]
    print(qsort(seq))
```

### 7.12 找零问题

```python
def  coinChange(values, money, coinsUsed):
    #values    T[1:n]数组
    #valuesCounts   钱币对应的种类数
    #money  找出来的总钱数
    #coinsUsed   对应于目前钱币总数i所使用的硬币数目
    for cents in range(1, money+1):
        minCoins = cents     #从第一个开始到money的所有情况初始
        for value in values:
            if value <= cents:
                temp = coinsUsed[cents - value] + 1
                if temp < minCoins:
                    minCoins = temp
        coinsUsed[cents] = minCoins
        print('面值为：{0} 的最小硬币数目为：{1} '.format(cents, coinsUsed[cents]) )
 
if __name__ == '__main__':
    values = [ 25, 21, 10, 5, 1]
    money = 63
    coinsUsed = {i:0 for i in range(money+1)}
    coinChange(values, money, coinsUsed)
```

### 7.13 广度遍历和深度遍历二叉树

给定一个数组，构建二叉树，并且按层次打印这个二叉树

```python
## 14 二叉树节点
class Node(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
 
tree = Node(1, Node(3, Node(7, Node(0)), Node(6)), Node(2, Node(5), Node(4)))
 
## 15 层次遍历
def lookup(root):
    stack = [root]
    while stack:
        current = stack.pop(0)
        print current.data
        if current.left:
            stack.append(current.left)
        if current.right:
            stack.append(current.right)
## 16 深度遍历
def deep(root):
    if not root:
        return
    print root.data
    deep(root.left)
    deep(root.right)
 
if __name__ == '__main__':
    lookup(tree)
    deep(tree)
```

### 7.14 前中后序遍历

深度遍历改变顺序就OK了

### 7.15 求最大树深

```python
def maxDepth(root):
        if not root:
            return 0
        return max(maxDepth(root.left), maxDepth(root.right)) + 1
```

### 7.16 求两棵树是否相同

```python
def isSameTree(p, q):
    if p == None and q == None:
        return True
    elif p and q :
        return p.val == q.val and isSameTree(p.left,q.left) and isSameTree(p.right,q.right)
    else :
        return False
```

### 7.17 前序中序求后序

推荐: http://blog.csdn.net/hinyunsin/article/details/6315502

```python
def rebuild(pre, center):
    if not pre:
        return
    cur = Node(pre[0])
    index = center.index(pre[0])
    cur.left = rebuild(pre[1:index + 1], center[:index])
    cur.right = rebuild(pre[index + 1:], center[index + 1:])
    return cur
 
def deep(root):
    if not root:
        return
    deep(root.left)
    deep(root.right)
    print root.data
```

### 7.18 单链表逆置

```python
class Node(object):
    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next
 
link = Node(1, Node(2, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9)))))))))
 
def rev(link):
    pre = link
    cur = link.next
    pre.next = None
    while cur:
        tmp = cur.next
        cur.next = pre
        pre = cur
        cur = tmp
    return pre
 
root = rev(link)
while root:
    print root.data
    root = root.next
```

### 7.19 求两个数字之间的素数

- 素数：只能被 1 及自己整除的数，如 3，7，13，23 等

```python
for i in range(6,33+1):
    for j in range(2,i+1):
        if i%j==0 and j
```

### 7.20 请用Python手写实现冒泡排序

解析：

冒泡排序的原理不难，假定要将被排序的数组 R[1..n] 从大到小垂直排列，每个数字 R 可以看作是重量为 R.key 的气泡。

根据轻气泡在上、重气泡在上的原则，从下往上扫描数组 R: 凡扫描到违反本原则的轻气泡，则使其向上"飘浮"。如此反复进行，直到最后任何两个气泡都是轻者在上、重者在下为止。
然后将所有气泡逆序，就实现了数组从小到大的排序。

步骤：

1. 比较相邻的元素。如果第一个比第二个大，就交换他们两个。
2. 对第 0 个到第 n-1 个数据做同样的工作。这时，最大的数就到了数组最后的位置上。
3. 针对所有的元素重复以上的步骤，除了最后一个。
4. 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。

```python
def bubble_sort(arry):
    #获得数组的长度
    n = len(arry)                   
    for i in range(n):
        for j in range(1,n-i):
            #如果前者比后者大
            if  arry[j-1] > arry[j] :  
                #则交换两者     
                arry[j-1],arry[j] = arry[j],arry[j-1]      
    return arry
```

### 7.21 请用Python手写实现选择排序

解析：

选择排序(Selection sort)是一种简单直观的排序算法。

它的工作原理如下。首先在未排序序列中找到最小元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小元素，然后放到排序序列第二个位置。以此类推，直到所有元素均排序完毕。

```python
def select_sort(ary):
    n = len(ary)
    for i in range(0,n):
        #最小元素下标标记
        min = i                             
        for j in range(i+1,n):
            if ary[j] < ary[min] :
                #找到最小值的下标
                min = j
        #交换两者                     
        ary[min],ary[i] = ary[i],ary[min]   
    return ary
```

### 7.22 请用Python手写实现插入排序

插入排序（Insertion Sort）的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。

算法执行步骤：

1. 从第一个元素开始，该元素可以认为已经被排序
2. 取出下一个元素，在已经排序的元素序列中从后向前扫描
3. 如果被扫描的元素（已排序）大于新元素，则将被扫描元素后移一位
4. 重复步骤 3，直到找到已排序的元素小于或者等于新元素的位置
5. 将新元素插入到该位置后
6. 重复步骤 2~5

<img src="_asset/插入排序.gif">

```python
def insert_sort(ary):
    n = len(ary)
    for i in range(1,n):
        if ary[i] < ary[i-1]:
            temp = ary[i]

            #待插入的下标
            index = i           
            #从i-1 循环到 0 (包括0)
            for j in range(i-1,-1,-1):  
                if ary[j] > temp :
                    ary[j+1] = ary[j]
                    #记录待插入下标
                    index = j   
                else :
                    break
            ary[index] = temp
    return ary
```

### 7.23 请用Python手写实现快速排序

解析：

步骤：

1. 从数列中挑出一个元素，称为 “基准”（pivot），
2. 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。
3. 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。

换言之

快速排序时基于分治模式处理的，

对一个典型子数组 A[p...r] 排序的分治过程为三个步骤：

1. 分解：

   A[p..r] 被划分为俩个（可能空）的子数组 A[p ..q-1] 和 A[q+1 ..r]，使得

   A[p ..q-1] <= A[q] <= A[q+1 ..r]

2. 解决：通过递归调用快速排序，对子数组 A[p ..q-1] 和 A[q+1 ..r] 排序。

3. 合并。

```python
QUICKSORT(A, p, r)
1 if p < r
2    then q ← PARTITION(A, p, r)   //关键
3         QUICKSORT(A, p, q - 1)
4         QUICKSORT(A, q + 1, r)
```

数组划分

快速排序算法的关键是 PARTITION 过程，它对 A[p..r] 进行就地重排：

```python
PARTITION(A, p, r)
1  x ← A[r]
2  i ← p - 1
3  for j ← p to r - 1
4       do if A[j] ≤ x
5             then i ← i + 1
6                  exchange A[i] <-> A[j]
7  exchange A[i + 1] <-> A[r]
8  return i + 1
```

下图是一个例子

<img src="_asset/快速排序-01.gif">

这是另外一个可视化图

<img src="_asset/快速排序-02.gif">

```python
def quick_sort(ary):
    return qsort(ary,0,len(ary)-1)

def qsort(ary,left,right):
    #快排函数，ary为待排序数组，left为待排序的左边界，right为右边界
    if left >= right : return ary
    key = ary[left]     #取最左边的为基准数
    lp = left           #左指针
    rp = right          #右指针

    while lp < rp :
        while ary[rp] >= key and lp < rp :
            rp -= 1
        while ary[lp] <= key and lp < rp :
            lp += 1
        ary[lp],ary[rp] = ary[rp],ary[lp]
    ary[left],ary[lp] = ary[lp],ary[left]
    qsort(ary,left,lp-1)
    qsort(ary,rp+1,right)
    return ary
```

### 7.24 请用Python手写实现堆排序

解析：

堆排序在 top K 问题中使用比较频繁。堆排序是采用二叉堆的数据结构来实现的，虽然实质上还是一维数组。

二叉堆是一个近似完全二叉树 。

二叉堆具有以下性质：

- 父节点的键值总是大于或等于（小于或等于）任何一个子节点的键值。
- 每个节点的左右子树都是一个二叉堆（都是最大堆或最小堆）。

步骤：

1. 构造最大堆（Build_Max_Heap）：若数组下标范围为 0~n，考虑到单独一个元素是大根堆，则从下标 n/2 开始的元素均为大根堆。于是只要从 n/2-1 开始，向前依次构造大根堆，这样就能保证，构造到某个节点时，它的左右子树都已经是大根堆。

2. 堆排序（HeapSort）：由于堆是用数组模拟的。得到一个大根堆后，数组内部并不是有序的。因此需要将堆化数组有序化。思想是移除根节点，并做最大堆调整的递归运算。第一次将 heap[0] 与 heap[n-1] 交换，再对 heap[0...n-2] 做最大堆调整。第二次将 heap[0] 与 heap[n-2] 交换，再对 heap[0...n-3] 做最大堆调整。重复该操作直至 heap[0] 和 heap[1] 交换。由于每次都是将最大的数并入到后面的有序区间，故操作完后整个数组就是有序的了。

3. 最大堆调整（Max_Heapify）：该方法是提供给上述两个过程调用的。目的是将堆的末端子节点作调整，使得子节点永远小于父节点。

<img src="_asset/堆排序.gif">

```python
def heap_sort(ary) :
    n = len(ary)
    #最后一个非叶子节点
    first = int(n/2-1)       
    #构造大根堆
    for start in range(first,-1,-1) :     
        max_heapify(ary,start,n-1)

    #堆排，将大根堆转换成有序数组
    for end in range(n-1,0,-1):           
        ary[end],ary[0] = ary[0],ary[end]
        max_heapify(ary,0,end-1)
    return ary

#最大堆调整：将堆的末端子节点作调整，使得子节点永远小于父节点
#start为当前需要调整最大堆的位置，end为调整边界
def max_heapify(ary,start,end):
    root = start
    while True :
        #调整节点的子节点
        child = root*2 +1               
        if child > end : break
        if child+1 <= end and ary[child] < ary[child+1] :
            #取较大的子节点
            child = child+1
        #较大的子节点成为父节点             
        if ary[root] < ary[child] :    
            #交换 
            ary[root],ary[child] = ary[child],ary[root]     
            root = child
        else :
            break
```

### 7.25 请用Python手写实现归并排序

解析：

归并排序是采用分治法的一个非常典型的应用。归并排序的思想就是先递归分解数组，再合并数组。

先考虑合并两个有序数组，基本思路是比较两个数组的最前面的数，谁小就先取谁，取了后相应的指针就往后移一位。然后再比较，直至一个数组为空，最后把另一个数组的剩余部分复制过来即可。

再考虑递归分解，基本思路是将数组分解成 left 和 right，如果这两个数组内部数据是有序的，那么就可以用上面合并数组的方法将这两个数组合并排序。如何让这两个数组内部是有序的？可以再二分，直至分解出的小组只含有一个元素时为止，此时认为该小组内部已有序。然后合并排序相邻二个小组即可。

<img src="_asset/归并排序.gif">

```python
def merge_sort(ary):
    if len(ary) <= 1 : return ary
    num = int(len(ary)/2)       #二分分解
    left = merge_sort(ary[:num])
    right = merge_sort(ary[num:])
    return merge(left,right)    #合并数组

def merge(left,right):
    '''合并操作，
    将两个有序数组left[]和right[]合并成一个大的有序数组'''
    #left与right数组的下标指针
    l,r = 0,0           
    result = []
    while l
```

### 7.26 链表操作

```python
# 链表操作
# 节点类
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        
node = Node(4)

# 链表类
class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        
    # is_empty方法检查链表是否是一个空链表，这个方法只需要检查head节点是否指向None即可
    # 如果是空列表返回True，否则返回False
    def is_empty(self):
        return self.head is None
    
    # append方法表示增加元素到链表，这和insert方法不同，前者使新增加的元素成为链表中第一个节点，
    # 而后者是根据索引值来判断插入到链表的哪个位置
    def append(self, data):
        node = Node(data)
        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node
            
    # iter方法表示遍历链表。在遍历链表时也要首先考虑空链表的情况。遍历链表时从head开始，
    # 直到一个节点的next指向None结束，需要考虑的情况：
    # 1. 空链表时
    # 2. 插入位置超出链表节点的长度时
    # 3. 插入位置是链表的最后一个节点时，需要移动tail
    def iter(self):
        if not self.head:
            return
        cur = self.head
        yield cur.data
        while cur.next:
            cur = cur.next
            yield cur.data
            
    # insert方法实现
    def insert(self, idx, value):
        cur = self.head
        cur_idx = 0
        if cur is None:
            raise Exception('The list is an empty list')
        while cur_idx < idx - 1:
            cur = cur.next
            if cur is None:
                raise Exception('List length less than index')
            cur_idx += 1
        node = Node(value)
        node.next = cur.next
        cur.next = node
        if node.next is None:
            self.tail = node
      
    # remove方法接收一个idx参数，表示要删除节点的索引，此方法要考虑以下几种情况：
    # 1. 空链表，直接抛出异常
    # 2. 删除第一个节点时，移动head到删除节点的next指针指向的对象
    # 3. 链表只有一个节点时，把head与tail都指向None即可
    # 4. 删除最后一个节点时，需要移动tail到上一个节点
    # 5. 遍历链表时要判断给定的索引是否大于链表的长度，如果大于则抛出异常信息
    def remove(self, idx):
        cur = self.head
        cur_idx = 0
        if self.head is None:  # 空链表时
            raise Exception('The list is an empty list')
        while cur_idx < idx - 1:
            cur = cur.next
            if cur is None:
                raise Exception('List length less than index')
            cur_idx += 1
        if idx == 0: # 当删除第一个节点时
            self.head = cur.next
            cur = cur.next
            return
        if self.head is self.tail: # 当前只有一个节点的链表时
            self.head = None
            self.tail = None
            return
        cur.next = cur.next.next
        if cur.next is None: # 当删除的节点是链表最后一个节点时
            self.tail = cur
    
    # size函数不接收参数，返回链表中节点的个数，要获得链表的节点个数，必定会遍历链表，
    # 直到最后一个节点的next指针指向None时链表遍历完成，遍历时可以用一个累加器来计算节点的个数
    def size(self):
        current = self.head
        count = 0
        if current is None:
            return 'The list is an empty list'
        while current is not None:
            count += 1
            current = current.next
        return count
    
    # search函数接收一个item参数，表示查找节点中数据域的值。search函数遍历链表，每到一个节点把当
    # 前节点的data值与item作比较，最糟糕的情况下会全遍历链表。如果查找到有些数据则返回True，否则返回False
    def search(self, item):
        current = self.head
        found = False
        while current is not None and not found:
            if current.data == item:
                found = True
            else:
                current = current.next
        return found
    
if __name__ == '__main__':
    link_list = LinkedList()
    for i in range(150):
        link_list.append(i)
    print(link_list.is_empty())
    link_list.insert(10, 30)
    link_list.remove(0)
    
    for node in link_list.iter():
        print('node is {0}'.format(node))
    
    print(link_list.size())
    print(link_list.search(20))
```



## Reference

> [很全的 Python 面试题](http://python.jobbole.com/85231/) - 伯乐在线

